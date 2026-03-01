"""Web UI for interacting with agents using Flask.

Run with ``python -m agent_test.ui`` after installing ``flask``.
"""

from __future__ import annotations

import json as _json
import os
import pathlib as _pathlib
import uuid as _uuid

from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context, url_for

from agent_test.agents.crewai_agent import CrewAIAgent, DEFAULT_MODEL
from agent_test.agents.resume.agent import (
    DEFAULT_MODEL as RESUME_DEFAULT_MODEL,
    ResumeAgent,
)

_TITLE_MAX = 30        # chars used as auto-title from first user message
_TITLE_RENAME_MAX = 60  # max chars allowed when manually renaming

_AGENT_TYPES = {
    "chat":   {"label": "General Chat",    "model": DEFAULT_MODEL},
    "resume": {"label": "Resume Analyzer", "model": RESUME_DEFAULT_MODEL},
}

# Server-side cache of compiled agents, keyed by session id.
# Each entry is evicted when its session is deleted or the server restarts.
_agent_cache: dict[str, CrewAIAgent | ResumeAgent] = {}

# Conversation histories stored server-side to avoid Flask's 4 KB cookie
# limit — HTML fit reports can be several kilobytes.  Persisted to disk so
# histories survive server restarts (e.g. debug-mode reloader).
_HISTORY_FILE = _pathlib.Path(__file__).parent.parent.parent / "conversation_history.json"


def _load_history() -> dict[str, list[dict[str, str]]]:
    """Load persisted histories from disk, returning empty dict on any error."""
    try:
        if _HISTORY_FILE.exists():
            return _json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_history() -> None:
    """Write the current _history_store to disk, ignoring I/O errors."""
    try:
        _HISTORY_FILE.write_text(
            _json.dumps(_history_store, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
    except Exception:
        pass


_history_store: dict[str, list[dict[str, str]]] = _load_history()


def _get_or_create_agent(sid: str, agent_type: str = "chat") -> CrewAIAgent | ResumeAgent:
    """Return the cached agent for *sid*, creating one on first access."""
    if sid not in _agent_cache:
        if agent_type == "resume":
            _agent_cache[sid] = ResumeAgent()
        else:
            _agent_cache[sid] = CrewAIAgent()
    return _agent_cache[sid]


def _evict_agent(sid: str) -> None:
    """Remove the cached agent and history for *sid* if present."""
    _agent_cache.pop(sid, None)
    _history_store.pop(sid, None)
    _save_history()


def _new_sid() -> str:
    return _uuid.uuid4().hex


def _ensure_active(sess_map: dict, agent_type: str = "chat") -> str:
    """Return the active session id.

    If the current active session is invalid, switches to the most recently
    added existing session.  Only creates a new session when ``sess_map`` is
    completely empty.
    """
    active_id = session.get("active_id")
    if not active_id or active_id not in sess_map:
        if sess_map:
            # switch to the last (most recently added) existing session
            active_id = next(reversed(list(sess_map)))
            session["active_id"] = active_id
        else:
            active_id = _new_sid()
            sess_map[active_id] = {
                "title": "New conversation",
                "agent_type": agent_type,
            }
            session["sessions"] = sess_map
            session["active_id"] = active_id
    return active_id


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev")  # set SECRET_KEY in production

    @app.route("/")
    def index():
        sess_map: dict = session.setdefault("sessions", {})
        active_id = _ensure_active(sess_map)
        current = sess_map[active_id]
        agent_type = current.get("agent_type", "chat")
        if agent_type not in _AGENT_TYPES:
            agent_type = "chat"
            current["agent_type"] = agent_type
            session["sessions"] = sess_map
        return render_template(
            "chat.html",
            history=_history_store.get(active_id, []),
            sessions=sess_map,
            active_id=active_id,
            agent_types=_AGENT_TYPES,
            agent_type=agent_type,
            model=_AGENT_TYPES[agent_type]["model"],
        )

    @app.route("/chat", methods=("POST",))
    def chat():
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "empty prompt"}), 400

        sess_map: dict = session.setdefault("sessions", {})
        active_id = _ensure_active(sess_map)
        current = sess_map[active_id]

        history_list = _history_store.setdefault(active_id, [])
        history_list.append({"role": "user", "content": prompt})
        # name the session after the first user message
        if current["title"] == "New conversation":
            current["title"] = prompt[:_TITLE_MAX] + ("\u2026" if len(prompt) > _TITLE_MAX else "")

        # pass all prior turns as history so the model has full context
        history = history_list[:-1]
        agent_type = current.get("agent_type", "chat")
        if agent_type not in _AGENT_TYPES:
            agent_type = "chat"
        agent = _get_or_create_agent(active_id, agent_type)
        steps: list = []
        try:
            if hasattr(agent, "act_stream"):
                response = ""
                for event in agent.act_stream(prompt, history=history):
                    if event["type"] == "error":
                        raise RuntimeError(str(event["text"]))
                    elif event["type"] == "step":
                        steps.append({"icon": event.get("icon", ""), "label": event.get("label", "")})
                    elif event["type"] == "response":
                        response = event["text"]
                if not response:
                    raise RuntimeError("Agent produced no response.")
            elif hasattr(agent, "act_detailed"):
                response, steps = agent.act_detailed(prompt, history=history)
            else:
                response = agent.act(prompt, history=history)
        except Exception as exc:
            history_list.pop()  # remove the user message we optimistically added
            session["sessions"] = sess_map
            return jsonify({"error": f"Agent error: {exc}"}), 500
        history_list.append({"role": "assistant", "content": response})
        _save_history()

        session["sessions"] = sess_map
        return jsonify({"response": response, "steps": steps})

    @app.route("/chat/stream", methods=("POST",))
    def chat_stream():
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "empty prompt"}), 400

        sess_map: dict = session.setdefault("sessions", {})
        active_id = _ensure_active(sess_map)
        current = sess_map[active_id]

        if current["title"] == "New conversation":
            current["title"] = prompt[:_TITLE_MAX] + ("\u2026" if len(prompt) > _TITLE_MAX else "")
        session["sessions"] = sess_map  # commit title to cookie before streaming

        history_list = _history_store.setdefault(active_id, [])
        history_list.append({"role": "user", "content": prompt})
        history = history_list[:-1]

        agent_type = current.get("agent_type", "chat")
        if agent_type not in _AGENT_TYPES:
            agent_type = "chat"
        agent = _get_or_create_agent(active_id, agent_type)

        def generate():
            response = ""
            try:
                if hasattr(agent, "act_stream"):
                    for event in agent.act_stream(prompt, history=history):
                        yield f"data: {_json.dumps(event)}\n\n"
                        if event["type"] == "response":
                            response = event["text"]
                        elif event["type"] == "error":
                            history_list.pop()
                            _save_history()
                            return
                    if not response:
                        history_list.pop()
                        _save_history()
                        yield f"data: {_json.dumps({'type': 'error', 'text': 'Agent produced no response.'})}\n\n"
                        return
                else:
                    resp_text = agent.act(prompt, history=history)
                    response = resp_text
                    yield f"data: {_json.dumps({'type': 'response', 'text': resp_text})}\n\n"
                history_list.append({"role": "assistant", "content": response})
                _save_history()
            except Exception as exc:
                history_list.pop()
                _save_history()
                yield f"data: {_json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/sessions/new")
    def new_session():
        agent_type = request.args.get("agent", "chat")
        if agent_type not in _AGENT_TYPES:
            agent_type = "chat"
        sess_map: dict = session.setdefault("sessions", {})
        new_id = _new_sid()
        sess_map[new_id] = {
            "title": "New conversation",
            "agent_type": agent_type,
        }
        session["sessions"] = sess_map
        session["active_id"] = new_id
        # no agent cache entry to evict — it will be created lazily on first chat
        return redirect(url_for("index"))

    @app.route("/sessions/<sid>")
    def switch_session(sid: str):
        sess_map: dict = session.get("sessions", {})
        if sid in sess_map:
            session["active_id"] = sid
        return redirect(url_for("index"))

    @app.route("/sessions/<sid>/delete", methods=("POST",))
    def delete_session(sid: str):
        sess_map: dict = session.get("sessions", {})
        sess_map.pop(sid, None)
        session["sessions"] = sess_map
        if session.get("active_id") == sid:
            session.pop("active_id", None)
        _evict_agent(sid)
        # Tell the client where to navigate next:
        # - the most recently added remaining session, or
        # - None (client will create a fresh session explicitly)
        next_id = next(reversed(list(sess_map)), None)
        return jsonify({"ok": True, "next": next_id})

    @app.route("/sessions/<sid>/rename", methods=("POST",))
    def rename_session(sid: str):
        sess_map: dict = session.get("sessions", {})
        if sid in sess_map:
            data = request.get_json(force=True)
            title = (data.get("title") or "").strip()
            if title:
                sess_map[sid]["title"] = title[:_TITLE_RENAME_MAX]
                session["sessions"] = sess_map
        return jsonify({"ok": True})

    # kept for backwards-compat; now just opens a new session
    @app.route("/clear")
    def clear():
        return redirect(url_for("new_session"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
