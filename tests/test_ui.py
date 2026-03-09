"""Tests for the Flask web UI in :mod:`agent_test.ui`."""

from __future__ import annotations

import pytest

import agent_test.ui as ui_module
from agent_test.ui import create_app


@pytest.fixture(autouse=True)
def clear_agent_cache():
    """Wipe the module-level agent cache and history store before every test."""
    ui_module._agent_cache.clear()
    ui_module._history_store.clear()
    ui_module._pipeline_sessions.clear()
    yield
    ui_module._agent_cache.clear()
    ui_module._history_store.clear()
    ui_module._pipeline_sessions.clear()
    # Remove any history file written during the test to keep the workspace clean.
    ui_module._HISTORY_FILE.unlink(missing_ok=True)


def test_index_get() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Agent Chat" in resp.data
    assert b'<div class="msg-row' not in resp.data


def test_index_post_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAgent:
        def act(self, obs: str, history=None) -> str:
            return "dummy:" + obs

    dummy = DummyAgent()
    monkeypatch.setattr("agent_test.ui._get_or_create_agent", lambda sid, agent_type="chat": dummy)

    app = create_app()
    client = app.test_client()

    resp = client.post("/chat", json={"prompt": "hey"})
    assert resp.status_code == 200
    assert resp.get_json()["response"] == "dummy:hey"

    resp2 = client.get("/")
    assert b"hey" in resp2.data
    assert b"dummy:hey" in resp2.data

    client.post("/chat", json={"prompt": "again"})
    resp3 = client.get("/")
    assert b"hey" in resp3.data
    assert b"dummy:hey" in resp3.data
    assert b"again" in resp3.data
    assert b"dummy:again" in resp3.data


def test_clear_resets_history(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAgent:
        def act(self, obs: str, history=None) -> str:
            return "r:" + obs

    dummy = DummyAgent()
    monkeypatch.setattr("agent_test.ui._get_or_create_agent", lambda sid, agent_type="chat": dummy)

    app = create_app()
    client = app.test_client()

    client.post("/chat", json={"prompt": "hello"})
    client.get("/clear", follow_redirects=True)
    resp = client.get("/")
    assert b'<div class="msg-row' not in resp.data


def test_pipeline_page_get() -> None:
    app = create_app()
    client = app.test_client()

    resp = client.get("/pipeline")

    assert resp.status_code == 200
    assert b"Guided Resume Workflow" in resp.data
    assert b"Editable Improved Resume" in resp.data


def test_pipeline_reanalyze_accepts_resume_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class DummyPipelineOrchestrator:
        def stream_fit_analysis(self, resume_text: str, job_description: str):
            captured["resume_text"] = resume_text
            captured["job_description"] = job_description
            yield {"type": "result", "text": "<div class=\"fit-report\">Updated</div>"}

    monkeypatch.setattr("agent_test.ui.PipelineOrchestrator", DummyPipelineOrchestrator)

    app = create_app()
    client = app.test_client()

    with client.session_transaction() as flask_session:
      flask_session["pipeline_id"] = "pipeline-1"

    ui_module._pipeline_sessions["pipeline-1"] = {
        "resume_text": "original resume",
        "job_description": "job description",
        "fit_report": "initial report",
        "improved_resume": "old improved resume",
    }

    resp = client.post("/pipeline/reanalyze", json={"resume_text": "reviewed improved resume"})

    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert '"type": "result"' in body
    assert captured == {
        "resume_text": "reviewed improved resume",
        "job_description": "job description",
    }
    assert ui_module._pipeline_sessions["pipeline-1"]["improved_resume"] == "reviewed improved resume"
    assert ui_module._pipeline_sessions["pipeline-1"]["reanalyzed_fit_report"] == '<div class="fit-report">Updated</div>'

