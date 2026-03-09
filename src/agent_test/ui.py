"""Web UI for interacting with agents using Flask.

Run with ``python -m agent_test.ui`` after installing ``flask``.
"""

from __future__ import annotations

import json as _json
import os
import pathlib as _pathlib
import uuid as _uuid

from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context, url_for

from agent_test.config import DEFAULT_MODEL
from agent_test.agents.crewai_agent import CrewAIAgent
from agent_test.agents.fit_analyzer.agent import FitAnalyzerAgent
from agent_test.agents.resume_improve.agent import ResumeImproveAgent
from agent_test.agents.pipeline import PipelineOrchestrator

_TITLE_MAX = 30        # chars used as auto-title from first user message
_TITLE_RENAME_MAX = 60  # max chars allowed when manually renaming
_MAX_PIPELINE_SESSIONS = 12

_AGENT_TYPES = {
    "chat":            {"label": "General Chat",        "model": DEFAULT_MODEL},
    "resume":          {"label": "Resume Analyzer",     "model": DEFAULT_MODEL},
    "resume_improve":  {"label": "Resume Improvement",  "model": DEFAULT_MODEL},
}

# Server-side state for the pipeline page.  Each entry stores the inputs and
# outputs for one pipeline run, keyed by a pipeline-session UUID.
_pipeline_sessions: dict[str, dict] = {}

# Server-side cache of compiled agents, keyed by session id.
# Each entry is evicted when its session is deleted or the server restarts.
_agent_cache: dict[str, CrewAIAgent | FitAnalyzerAgent | ResumeImproveAgent] = {}

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


def _get_or_create_agent(
    sid: str,
    agent_type: str = "chat",
) -> CrewAIAgent | FitAnalyzerAgent | ResumeImproveAgent:
    """Return the cached agent for *sid*, creating one on first access."""
    if sid not in _agent_cache:
        if agent_type == "resume":
            _agent_cache[sid] = FitAnalyzerAgent()
        elif agent_type == "resume_improve":
            _agent_cache[sid] = ResumeImproveAgent()
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


def _evict_pipeline_session(pid: str | None) -> None:
    """Remove a pipeline session from the in-memory store and request session."""
    if not pid:
        return
    _pipeline_sessions.pop(pid, None)
    if session.get("pipeline_id") == pid:
        session.pop("pipeline_id", None)


def _reset_current_pipeline_session() -> None:
    """Drop the current pipeline session, if the request has one."""
    _evict_pipeline_session(session.get("pipeline_id"))


def _prune_pipeline_sessions(keep_pid: str | None = None) -> None:
    """Bound server-side pipeline memory usage.

    The refreshed pipeline UI encourages repeated staged runs, so prune old
    in-memory sessions and keep at most ``_MAX_PIPELINE_SESSIONS`` entries.
    """
    if len(_pipeline_sessions) <= _MAX_PIPELINE_SESSIONS:
        return

    for pid in list(_pipeline_sessions):
        if len(_pipeline_sessions) <= _MAX_PIPELINE_SESSIONS:
            break
        if pid == keep_pid:
            continue
        _pipeline_sessions.pop(pid, None)


def _get_current_pipeline_state() -> dict | None:
    """Return a JSON-safe copy of the current pipeline session state."""
    pid = session.get("pipeline_id")
    state = _pipeline_sessions.get(pid) if pid else None
    if not state:
        if pid:
            session.pop("pipeline_id", None)
        return None

    return {
        "pipeline_id": pid,
        "resume_text": state.get("resume_text", ""),
        "job_description": state.get("job_description", ""),
        "fit_report": state.get("fit_report"),
        "improvement_report": state.get("improvement_report"),
        "improved_resume": state.get("improved_resume"),
        "reanalyzed_fit_report": state.get("reanalyzed_fit_report"),
        "consent": state.get("consent", "Pending"),
    }


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

    # ── Pipeline ────────────────────────────────────────────────────

    @app.route("/pipeline")
    def pipeline_page():
        return render_template(
            "pipeline.html",
            model=DEFAULT_MODEL,
            pipeline_state=_get_current_pipeline_state(),
        )

    @app.route("/pipeline/reset", methods=("POST",))
    def pipeline_reset():
        _reset_current_pipeline_session()
        return jsonify({"ok": True})

    @app.route("/pipeline/analyze", methods=("POST",))
    def pipeline_analyze():
        data = request.get_json(force=True)
        resume_text = (data.get("resume_text") or "").strip()
        job_description = (data.get("job_description") or "").strip()
        if not resume_text or not job_description:
            return jsonify({"error": "Both resume and job description are required."}), 400

        _reset_current_pipeline_session()

        # Create a fresh pipeline session.
        pid = _uuid.uuid4().hex
        _pipeline_sessions[pid] = {
            "resume_text": resume_text,
            "job_description": job_description,
            "fit_report": None,
            "improvement_report": None,
            "improved_resume": None,
            "reanalyzed_fit_report": None,
            "consent": "Pending",
        }
        session["pipeline_id"] = pid
        _prune_pipeline_sessions(keep_pid=pid)

        orchestrator = PipelineOrchestrator()

        def generate():
            for event in orchestrator.stream_fit_analysis(resume_text, job_description):
                if event.get("type") == "result":
                    _pipeline_sessions[pid]["fit_report"] = event["text"]
                if event.get("type") != "heartbeat":
                    yield f"data: {_json.dumps(event)}\n\n"
                else:
                    yield ": heartbeat\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/pipeline/improve", methods=("POST",))
    def pipeline_improve():
        pid = session.get("pipeline_id")
        state = _pipeline_sessions.get(pid) if pid else None
        if not state or not state.get("fit_report"):
            return jsonify({"error": "No fit analysis found — run /pipeline/analyze first."}), 400

        orchestrator = PipelineOrchestrator()
        resume_text = state["resume_text"]
        job_description = state["job_description"]
        fit_report = state["fit_report"]

        def generate():
            for event in orchestrator.stream_improvement(resume_text, job_description, fit_report):
                if event.get("type") == "result":
                    _pipeline_sessions[pid]["improvement_report"] = event.get("text", "")
                    _pipeline_sessions[pid]["improved_resume"] = event.get("improved_resume", "")
                    _pipeline_sessions[pid]["consent"] = "Approved"
                if event.get("type") != "heartbeat":
                    yield f"data: {_json.dumps(event)}\n\n"
                else:
                    yield ": heartbeat\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/pipeline/reanalyze", methods=("POST",))
    def pipeline_reanalyze():
        pid = session.get("pipeline_id")
        state = _pipeline_sessions.get(pid) if pid else None
        if not state or not state.get("improved_resume"):
            return jsonify({"error": "No improved resume found — run /pipeline/improve first."}), 400

        data = request.get_json(silent=True) or {}
        improved_resume = (data.get("resume_text") or state.get("improved_resume") or "").strip()
        if not improved_resume:
            return jsonify({"error": "Improved resume text is required for re-analysis."}), 400
        state["improved_resume"] = improved_resume

        orchestrator = PipelineOrchestrator()
        job_description = state["job_description"]

        def generate():
            for event in orchestrator.stream_fit_analysis(improved_resume, job_description):
                if event.get("type") == "result":
                    _pipeline_sessions[pid]["reanalyzed_fit_report"] = event["text"]
                if event.get("type") != "heartbeat":
                    yield f"data: {_json.dumps(event)}\n\n"
                else:
                    yield ": heartbeat\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
