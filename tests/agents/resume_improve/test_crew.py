"""Tests for :func:`run_resume_improve_crew`.

CrewAI validation is bypassed by patching Agent, Task, and Crew so tests
run fast and fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crewai import Process

from agent_test.agents.resume_improve.crew import run_resume_improve_crew

_SAMPLE_RESUME = "Alice | SWE | 6 years Python Django FastAPI"
_SAMPLE_FIT = "JOB FIT ANALYSIS\nScore: 72%\nStrengths: Python\nGaps: AWS"
_SAMPLE_JD = "Senior Python Engineer 5 years FastAPI AWS required"


def _run_with_capture(fixed_llm, resume=_SAMPLE_RESUME, fit=_SAMPLE_FIT, jd=_SAMPLE_JD):
    """Run run_resume_improve_crew with CrewAgent, Task, and Crew fully mocked."""
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
            return "MOCK_ENHANCE_REPORT"

    with patch("agent_test.agents.resume_improve.crew.CrewAgent", side_effect=make_agent), \
         patch("agent_test.agents.resume_improve.crew.Task", side_effect=make_task), \
         patch("agent_test.agents.resume_improve.crew.Crew", CapturingCrew):
        result = run_resume_improve_crew(fixed_llm, resume, fit, jd)

    return result, captured


def test_returns_string(fixed_llm):
    result, _ = _run_with_capture(fixed_llm)
    assert isinstance(result, str)


def test_returns_crew_kickoff_output(fixed_llm):
    result, _ = _run_with_capture(fixed_llm)
    assert result == "MOCK_ENHANCE_REPORT"


def test_four_agents_created(fixed_llm):
    _, captured = _run_with_capture(fixed_llm)
    assert len(captured["agent_calls"]) == 4


def test_four_tasks_created(fixed_llm):
    _, captured = _run_with_capture(fixed_llm)
    assert len(captured["task_calls"]) == 4


def test_four_tasks_passed_to_crew(fixed_llm):
    _, captured = _run_with_capture(fixed_llm)
    assert len(captured["crew_tasks"]) == 4


def test_sequential_process(fixed_llm):
    _, captured = _run_with_capture(fixed_llm)
    assert captured["crew_kwargs"]["process"] == Process.sequential


def test_fit_analysis_in_first_task_description(fixed_llm):
    """Fit analysis text must appear in the first (strategy) task description."""
    _, captured = _run_with_capture(fixed_llm)
    assert _SAMPLE_FIT in captured["task_calls"][0]["description"]


def test_resume_text_in_second_task_description(fixed_llm):
    """Resume text must appear in the second (audit) task description."""
    _, captured = _run_with_capture(fixed_llm)
    assert _SAMPLE_RESUME in captured["task_calls"][1]["description"]


def test_agent_roles_are_unique(fixed_llm):
    _, captured = _run_with_capture(fixed_llm)
    roles = [c["role"] for c in captured["agent_calls"]]
    assert len(set(roles)) == 4
