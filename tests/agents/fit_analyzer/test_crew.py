"""Tests for :func:`run_resume_crew`.

CrewAI 0.100+ validates both ``Agent`` (requires a real LLM model string) and
``Task`` (requires a real BaseAgent instance) at construction time.  We patch
both to bypass pydantic validation, capturing constructor kwargs so we can
assert on the wiring without any live LLM or network calls.

What these tests verify
────────────────────────
* Public API — returns a plain string equal to crew.kickoff() output.
* Crew assembly — exactly 4 agents and 4 tasks, sequential process.
* Task description wiring — JD text in task[0], resume text in task[1].
* Agent roles — all four are unique, non-empty strings.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crewai import Process

from agent_test.agents.fit_analyzer.crew import run_resume_crew


_SAMPLE_JD = "Senior Python Engineer 5 years required FastAPI AWS"
_SAMPLE_RESUME = "John Doe 6 years Python Django Flask no AWS"


# ---------------------------------------------------------------------------
# Central capture helper
# ---------------------------------------------------------------------------

def _run_with_capture(fixed_llm, jd=_SAMPLE_JD, resume=_SAMPLE_RESUME):
    """Run run_resume_crew with CrewAgent, Task, and Crew fully mocked."""
    captured = {}
    agent_idx = [0]
    task_idx = [0]

    def make_agent(*args, **kwargs):
        i = agent_idx[0]; agent_idx[0] += 1
        captured.setdefault("agent_calls", []).append(kwargs)
        m = MagicMock(); m.role = kwargs.get("role", f"role_{i}")
        return m

    def make_task(*args, **kwargs):
        i = task_idx[0]; task_idx[0] += 1
        captured.setdefault("task_calls", []).append(kwargs)
        m = MagicMock(); m.description = kwargs.get("description", "")
        return m

    class CapturingCrew:
        def __init__(self, **kwargs):
            captured["crew_kwargs"] = kwargs
            captured["crew_tasks"] = kwargs.get("tasks", [])
        def kickoff(self):
            return "MOCK_REPORT"

    with patch("agent_test.agents.fit_analyzer.crew.CrewAgent", side_effect=make_agent), \
         patch("agent_test.agents.fit_analyzer.crew.Task", side_effect=make_task), \
         patch("agent_test.agents.fit_analyzer.crew.Crew", CapturingCrew):
        result = run_resume_crew(fixed_llm, jd, resume)

    return result, captured


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


def test_returns_string(fixed_llm):
    result, _ = _run_with_capture(fixed_llm)
    assert isinstance(result, str)


def test_returns_crew_kickoff_output(fixed_llm):
    result, _ = _run_with_capture(fixed_llm)
    assert result == "MOCK_REPORT"


# ---------------------------------------------------------------------------
# Crew assembly
# ---------------------------------------------------------------------------


def test_four_agents_created(fixed_llm):
    """Exactly four CrewAgent constructor calls must occur."""
    _, captured = _run_with_capture(fixed_llm)
    assert len(captured["agent_calls"]) == 4


def test_four_tasks_created(fixed_llm):
    """Exactly four Task constructor calls must occur."""
    _, captured = _run_with_capture(fixed_llm)
    assert len(captured["task_calls"]) == 4


def test_four_tasks_passed_to_crew(fixed_llm):
    """All four task mocks must be forwarded to the Crew constructor."""
    _, captured = _run_with_capture(fixed_llm)
    assert len(captured["crew_tasks"]) == 4


def test_sequential_process_passed_to_crew(fixed_llm):
    _, captured = _run_with_capture(fixed_llm)
    assert captured["crew_kwargs"]["process"] == Process.sequential


# ---------------------------------------------------------------------------
# Task description content
# ---------------------------------------------------------------------------


def test_jd_text_in_first_task_description(fixed_llm):
    """Job description text must be embedded in the first (parse) task."""
    _, captured = _run_with_capture(fixed_llm)
    assert _SAMPLE_JD in captured["task_calls"][0]["description"]


def test_resume_text_in_second_task_description(fixed_llm):
    """Resume text must be embedded in the second (analyze) task."""
    _, captured = _run_with_capture(fixed_llm)
    assert _SAMPLE_RESUME in captured["task_calls"][1]["description"]


def test_third_task_has_scoring_rubric(fixed_llm):
    """The scoring task description must mention the weighted rubric."""
    _, captured = _run_with_capture(fixed_llm)
    desc = captured["task_calls"][2]["description"]
    assert "40" in desc or "scoring" in desc.lower() or "%" in desc


def test_fourth_task_has_output_format(fixed_llm):
    """The report task description must reference the output format template."""
    _, captured = _run_with_capture(fixed_llm)
    desc = captured["task_calls"][3]["description"]
    # The output format section header or Recommendation section must appear.
    assert "RECOMMENDATION" in desc or "Action Plan" in desc or "FIT ANALYSIS" in desc


# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------


def test_agent_roles_are_unique(fixed_llm):
    """All four role strings must be distinct and non-empty."""
    _, captured = _run_with_capture(fixed_llm)
    roles = [kw["role"] for kw in captured["agent_calls"]]
    assert all(isinstance(r, str) and r for r in roles)
    assert len(roles) == len(set(roles)), "Duplicate agent roles detected"
