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
    yield
    ui_module._agent_cache.clear()
    ui_module._history_store.clear()
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

