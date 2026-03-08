"""Tests for input-parsing utilities inside the resume graph.

:func:`_extract_json` and :func:`_conversation_to_str` are private helpers
but contain critical logic — they are imported directly and tested as units.
"""

from __future__ import annotations

import json

import pytest

from agent_test.agents.fit_analyzer.graph import (
    _conversation_to_str,
    _extract_json,
    _route_after_input,
)
from agent_test.agents.fit_analyzer.state import FitAnalyzerState


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def _valid_payload(overrides: dict | None = None) -> dict:
    base = {
        "has_jd": True,
        "has_resume": True,
        "clarification_needed": False,
        "clarification_question": "",
        "job_description": "Some JD text",
        "resume_text": "Some resume text",
    }
    if overrides:
        base.update(overrides)
    return base


def test_extract_json_clean_string() -> None:
    """A bare JSON string is parsed correctly."""
    payload = _valid_payload()
    result = _extract_json(json.dumps(payload))
    assert result["has_jd"] is True
    assert result["job_description"] == "Some JD text"


def test_extract_json_with_markdown_fence() -> None:
    """JSON wrapped in ```json ... ``` fences is stripped and parsed."""
    payload = _valid_payload()
    fenced = f"```json\n{json.dumps(payload)}\n```"
    result = _extract_json(fenced)
    assert result["has_resume"] is True


def test_extract_json_with_plain_code_fence() -> None:
    """JSON wrapped in plain ``` fences (no language tag) is also handled."""
    payload = _valid_payload()
    fenced = f"```\n{json.dumps(payload)}\n```"
    result = _extract_json(fenced)
    assert result["resume_text"] == "Some resume text"


def test_extract_json_embedded_in_prose() -> None:
    """If the LLM returns prose + JSON block, the first {{ }} block is extracted."""
    payload = _valid_payload()
    prose = f"Here is my answer:\n{json.dumps(payload)}\nDone."
    result = _extract_json(prose)
    assert result["clarification_needed"] is False


def test_extract_json_fallback_on_garbage() -> None:
    """Completely unparseable input returns a safe fallback with clarification."""
    result = _extract_json("I am unable to respond in JSON format today!")
    assert result["clarification_needed"] is True
    assert isinstance(result["clarification_question"], str)
    assert result["job_description"] == ""
    assert result["resume_text"] == ""


def test_extract_json_clarification_needed_true() -> None:
    """When the LLM signals clarification_needed=true, the flag propagates."""
    payload = _valid_payload({
        "has_jd": False,
        "clarification_needed": True,
        "clarification_question": "Please paste the job description.",
        "job_description": "",
    })
    result = _extract_json(json.dumps(payload))
    assert result["clarification_needed"] is True
    assert result["clarification_question"] == "Please paste the job description."


# ---------------------------------------------------------------------------
# _conversation_to_str
# ---------------------------------------------------------------------------


def test_conversation_to_str_single_message() -> None:
    messages = [{"role": "user", "content": "hello there"}]
    text = _conversation_to_str(messages)
    assert "User" in text
    assert "hello there" in text


def test_conversation_to_str_multiple_messages() -> None:
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]
    text = _conversation_to_str(messages)
    assert "question" in text
    assert "answer" in text
    assert "follow-up" in text


def test_conversation_to_str_role_capitalised() -> None:
    messages = [{"role": "user", "content": "msg"}]
    text = _conversation_to_str(messages)
    assert "User:" in text


def test_conversation_to_str_empty_list() -> None:
    assert _conversation_to_str([]) == ""


# ---------------------------------------------------------------------------
# _route_after_input routing logic
# ---------------------------------------------------------------------------


def _state_with(clarification_needed: bool) -> FitAnalyzerState:
    return FitAnalyzerState(
        messages=[],
        job_description="jd" if not clarification_needed else "",
        resume_text="cv" if not clarification_needed else "",
        clarification_needed=clarification_needed,
        clarification_question="Please provide JD." if clarification_needed else "",
        analysis_result="",
        response="",
    )


def test_route_to_ask_user_when_clarification_needed() -> None:
    assert _route_after_input(_state_with(clarification_needed=True)) == "ask_user"


def test_route_to_crew_when_both_inputs_present() -> None:
    assert _route_after_input(_state_with(clarification_needed=False)) == "crew_node"
