"""Tests for :class:`ResumeImproveState`.

Verifies the TypedDict schema has all required keys.
"""

from __future__ import annotations

from agent_test.agents.resume_improve.state import ResumeImproveState

_REQUIRED_KEYS = {
    "messages",
    "resume_text",
    "fit_analysis",
    "job_description",
    "clarification_needed",
    "clarification_question",
    "enhancement_result",
    "response",
}


def test_all_required_keys_present() -> None:
    """ResumeImproveState must declare all eight required fields."""
    assert _REQUIRED_KEYS == set(ResumeImproveState.__annotations__.keys())


def test_valid_state_construction() -> None:
    state: ResumeImproveState = {
        "messages": [{"role": "user", "content": "hi"}],
        "resume_text": "Alice | SWE",
        "fit_analysis": "Score: 72%",
        "job_description": "Senior SWE",
        "clarification_needed": False,
        "clarification_question": "",
        "enhancement_result": "",
        "response": "",
    }
    assert state["clarification_needed"] is False
    assert state["resume_text"] == "Alice | SWE"
