"""Tests for :class:`ResumeImproveAgent`.

Verifies end-to-end behaviour through ``act()`` using injected stubs.
The CrewAI crew is monkeypatched so tests remain fast and offline.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent_test.agents.resume_improve.agent import DEFAULT_MODEL, ResumeImproveAgent
from agent_test.agents.resume_improve.graph import _GREETING
from tests.conftest import JsonLLM

# Simulate a prior assistant turn so the first-turn short-circuit is bypassed.
_PRIOR_TURN = [{"role": "assistant", "content": _GREETING}]
_MOCK_REPORT = '<div class="improve-report">MOCK ENHANCEMENT REPORT</div>'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_that_requests_clarification(question: str = "Please paste your resume.") -> JsonLLM:
    payload = {
        "has_resume": False,
        "has_fit_analysis": False,
        "has_job_description": False,
        "clarification_needed": True,
        "clarification_question": question,
        "resume_text": "",
        "fit_analysis": "",
        "job_description": "",
    }
    return JsonLLM(json_response=json.dumps(payload))


def _llm_that_has_all_inputs(
    resume: str = "Resume text",
    fit: str = "Fit analysis text",
    jd: str = "Job description text",
) -> JsonLLM:
    payload = {
        "has_resume": True,
        "has_fit_analysis": True,
        "has_job_description": True,
        "clarification_needed": False,
        "clarification_question": "",
        "resume_text": resume,
        "fit_analysis": fit,
        "job_description": jd,
    }
    return JsonLLM(json_response=json.dumps(payload))


# ---------------------------------------------------------------------------
# Clarification path
# ---------------------------------------------------------------------------


def test_act_returns_greeting_on_first_turn() -> None:
    """On the very first turn the agent returns the welcome greeting."""
    agent = ResumeImproveAgent(llm=_llm_that_requests_clarification())
    result = agent.act("hi")
    assert result == _GREETING


def test_act_returns_clarification_when_missing_resume() -> None:
    """When resume is missing, the LLM returns a clarification question."""
    question = "Please paste your full resume."
    llm = _llm_that_requests_clarification(question)
    agent = ResumeImproveAgent(llm=llm)
    result = agent.act("Here is my fit analysis", history=_PRIOR_TURN)
    assert result == question


def test_act_clarification_does_not_call_crew() -> None:
    """The CrewAI pipeline must NOT run when clarification is needed."""
    llm = _llm_that_requests_clarification()
    agent = ResumeImproveAgent(llm=llm)

    with patch("agent_test.agents.resume_improve.graph.run_resume_improve_crew") as mock_crew:
        agent.act("partial input")
        mock_crew.assert_not_called()


# ---------------------------------------------------------------------------
# Analysis path
# ---------------------------------------------------------------------------


def test_act_returns_report_when_all_inputs_present() -> None:
    """When all inputs are extracted, the crew report is returned."""
    llm = _llm_that_has_all_inputs()
    agent = ResumeImproveAgent(llm=llm)

    with patch(
        "agent_test.agents.resume_improve.graph.run_resume_improve_crew",
        return_value=_MOCK_REPORT,
    ):
        result = agent.act("Here is everything: ...", history=_PRIOR_TURN)

    assert result == _MOCK_REPORT


def test_act_passes_inputs_to_crew() -> None:
    """Extracted resume, fit analysis, and JD are forwarded to run_resume_improve_crew."""
    resume = "Alice | SWE | Python, FastAPI"
    fit = "JOB FIT ANALYSIS\nScore: 75%"
    jd = "Senior Python Engineer"
    llm = _llm_that_has_all_inputs(resume=resume, fit=fit, jd=jd)
    agent = ResumeImproveAgent(llm=llm)

    with patch(
        "agent_test.agents.resume_improve.graph.run_resume_improve_crew",
        return_value=_MOCK_REPORT,
    ) as mock_crew:
        agent.act("All inputs here", history=_PRIOR_TURN)

    call_kwargs = mock_crew.call_args
    assert call_kwargs.kwargs.get("resume_text") == resume or resume in call_kwargs.args
    assert call_kwargs.kwargs.get("fit_analysis") == fit or fit in call_kwargs.args


def test_act_result_is_string() -> None:
    """act() must always return a plain string."""
    llm = _llm_that_has_all_inputs()
    agent = ResumeImproveAgent(llm=llm)

    with patch(
        "agent_test.agents.resume_improve.graph.run_resume_improve_crew",
        return_value=_MOCK_REPORT,
    ):
        result = agent.act("All inputs")

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Instance attributes
# ---------------------------------------------------------------------------


def test_model_attribute_stored() -> None:
    llm = _llm_that_requests_clarification()
    agent = ResumeImproveAgent(llm=llm, model="my/model")
    assert agent.model == "my/model"


def test_default_model_is_non_empty_string() -> None:
    assert isinstance(DEFAULT_MODEL, str) and DEFAULT_MODEL


def test_temperature_attribute_stored() -> None:
    llm = _llm_that_requests_clarification()
    agent = ResumeImproveAgent(llm=llm, temperature=0.2)
    assert agent.temperature == 0.2


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating ResumeImproveAgent without API key (and no injected llm) raises."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "agent_test.utils.openrouter_client.load_dotenv", lambda *_: False
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ResumeImproveAgent()
