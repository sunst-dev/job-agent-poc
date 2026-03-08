"""Tests for :class:`ResumeAgent`.

Verifies end-to-end behaviour through ``act()`` using injected stubs.
The CrewAI crew is monkeypatched so tests remain fast and offline.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent_test.agents.fit_analyzer.agent import DEFAULT_MODEL, ResumeAgent
from agent_test.agents.fit_analyzer.graph import _GREETING
from tests.conftest import JsonLLM

# Simulate a prior assistant turn so the first-turn short-circuit is bypassed
# and the injected LLM stub is actually invoked.
_PRIOR_TURN = [{"role": "assistant", "content": _GREETING}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_that_requests_clarification(question: str = "Please paste the JD.") -> JsonLLM:
    """Return a JsonLLM that signals the input_collector to ask the user."""
    payload = {
        "has_jd": False,
        "has_resume": False,
        "clarification_needed": True,
        "clarification_question": question,
        "job_description": "",
        "resume_text": "",
    }
    return JsonLLM(json_response=json.dumps(payload))


def _llm_that_has_both_inputs(jd: str = "JD text", resume: str = "Resume text") -> JsonLLM:
    """Return a JsonLLM that signals the input_collector both inputs are ready."""
    payload = {
        "has_jd": True,
        "has_resume": True,
        "clarification_needed": False,
        "clarification_question": "",
        "job_description": jd,
        "resume_text": resume,
    }
    return JsonLLM(json_response=json.dumps(payload))


_MOCK_REPORT = "🎯 JOB FIT ANALYSIS\nOVERALL FIT SCORE: 82%"


# ---------------------------------------------------------------------------
# Clarification path
# ---------------------------------------------------------------------------


def test_act_returns_clarification_when_no_inputs() -> None:
    """On the very first turn the agent always returns the welcome greeting."""
    agent = ResumeAgent(llm=_llm_that_requests_clarification())  # LLM not called
    result = agent.act("hi")
    assert result == _GREETING


def test_act_returns_clarification_when_missing_resume() -> None:
    """When JD is present but resume is missing, the LLM returns a clarification."""
    question = "Please also paste your resume or qualifications summary."
    llm = _llm_that_requests_clarification(question)
    agent = ResumeAgent(llm=llm)
    result = agent.act("Here is the job description: <JD text>", history=_PRIOR_TURN)
    assert result == question


def test_act_clarification_does_not_call_crew() -> None:
    """The CrewAI pipeline must NOT run when clarification is needed."""
    llm = _llm_that_requests_clarification()
    agent = ResumeAgent(llm=llm)

    with patch("agent_test.agents.fit_analyzer.graph.run_resume_crew") as mock_crew:
        agent.act("partial input")
        mock_crew.assert_not_called()


# ---------------------------------------------------------------------------
# Analysis path
# ---------------------------------------------------------------------------


def test_act_returns_report_when_both_inputs_present() -> None:
    """When JD and resume are both extracted, the crew report is returned."""
    llm = _llm_that_has_both_inputs()
    agent = ResumeAgent(llm=llm)

    with patch(
        "agent_test.agents.fit_analyzer.graph.run_resume_crew", return_value=_MOCK_REPORT
    ):
        result = agent.act("Here is my JD and resume: ...", history=_PRIOR_TURN)

    assert result == _MOCK_REPORT


def test_act_passes_jd_and_resume_to_crew() -> None:
    """The extracted JD and resume texts are forwarded to run_resume_crew."""
    jd = "Senior ML Engineer — PyTorch required"
    resume = "Alice | ML Engineer | TensorFlow, Keras"
    llm = _llm_that_has_both_inputs(jd=jd, resume=resume)
    agent = ResumeAgent(llm=llm)

    with patch(
        "agent_test.agents.fit_analyzer.graph.run_resume_crew", return_value=_MOCK_REPORT
    ) as mock_crew:
        agent.act("JD and resume pasted here", history=_PRIOR_TURN)

    call_kwargs = mock_crew.call_args
    assert call_kwargs.kwargs.get("job_description") == jd or call_kwargs.args[1] == jd
    assert call_kwargs.kwargs.get("resume_text") == resume or call_kwargs.args[2] == resume


def test_act_result_is_string_on_analysis_path() -> None:
    """act() must always return a plain string, even on the analysis path."""
    llm = _llm_that_has_both_inputs()
    agent = ResumeAgent(llm=llm)

    with patch(
        "agent_test.agents.fit_analyzer.graph.run_resume_crew", return_value=_MOCK_REPORT
    ):
        result = agent.act("JD + resume")

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# History handling
# ---------------------------------------------------------------------------


def test_act_history_is_threaded_through() -> None:
    """History passed to act() is included in the conversation state."""
    llm = _llm_that_requests_clarification("Paste JD please.")
    agent = ResumeAgent(llm=llm)
    history = [
        {"role": "user", "content": "previous turn"},
        {"role": "assistant", "content": "clarification reply"},
    ]
    # Should not crash and must return a string regardless of history content.
    result = agent.act("follow-up", history=history)
    assert isinstance(result, str)


def test_act_none_history_is_safe() -> None:
    """Passing history=None must not raise."""
    llm = _llm_that_requests_clarification()
    agent = ResumeAgent(llm=llm)
    result = agent.act("hello", history=None)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Instance attributes
# ---------------------------------------------------------------------------


def test_model_attribute_stored() -> None:
    """The model name is accessible on the agent instance."""
    llm = _llm_that_requests_clarification()
    agent = ResumeAgent(llm=llm, model="my/model")
    assert agent.model == "my/model"


def test_default_model_is_non_empty_string() -> None:
    assert isinstance(DEFAULT_MODEL, str) and DEFAULT_MODEL


def test_temperature_attribute_stored() -> None:
    llm = _llm_that_requests_clarification()
    agent = ResumeAgent(llm=llm, temperature=0.3)
    assert agent.temperature == 0.3


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating ResumeAgent without API key (and no injected llm) raises."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "agent_test.utils.openrouter_client.load_dotenv", lambda *_: False
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ResumeAgent()
