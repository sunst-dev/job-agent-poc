"""Tests for :class:`ResumeState`.

Verifies that the TypedDict schema has all required keys and that a
correctly shaped dict passes type validation at runtime.
"""

from __future__ import annotations

from agent_test.agents.resume.state import ResumeState


_FULL_STATE: ResumeState = {
    "messages": [{"role": "user", "content": "hi"}],
    "job_description": "Senior SWE at Acme",
    "resume_text": "5 years Python experience",
    "clarification_needed": False,
    "clarification_question": "",
    "analysis_result": "",
    "response": "",
}

_REQUIRED_KEYS = {
    "messages",
    "job_description",
    "resume_text",
    "clarification_needed",
    "clarification_question",
    "analysis_result",
    "response",
}


def test_all_required_keys_present() -> None:
    """ResumeState must declare all seven required fields."""
    assert _REQUIRED_KEYS == set(ResumeState.__annotations__.keys())


def test_messages_is_list() -> None:
    assert isinstance(_FULL_STATE["messages"], list)


def test_clarification_needed_is_bool() -> None:
    assert isinstance(_FULL_STATE["clarification_needed"], bool)


def test_string_fields_are_str() -> None:
    for key in ("job_description", "resume_text", "clarification_question",
                "analysis_result", "response"):
        assert isinstance(_FULL_STATE[key], str), f"{key!r} must be str"


def test_empty_strings_are_valid_defaults() -> None:
    """Fields may be empty strings — that signals 'not yet populated'."""
    state: ResumeState = {
        "messages": [],
        "job_description": "",
        "resume_text": "",
        "clarification_needed": True,
        "clarification_question": "",
        "analysis_result": "",
        "response": "",
    }
    assert state["clarification_needed"] is True
    assert state["job_description"] == ""
