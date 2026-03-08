"""Cross-agent smoke tests.

Verify that every public agent class is importable, exposes an ``act()``
method, and can be constructed with an injected LLM stub.  Detailed
behavioural tests live under ``tests/agents/``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_crewai_agent_importable() -> None:
    from agent_test.agents.crewai_agent import CrewAIAgent
    assert callable(CrewAIAgent)


def test_fit_analyzer_agent_importable() -> None:
    from agent_test.agents.fit_analyzer.agent import FitAnalyzerAgent
    assert callable(FitAnalyzerAgent)


# ---------------------------------------------------------------------------
# act() interface contract
# ---------------------------------------------------------------------------


def test_fit_analyzer_agent_act_is_callable() -> None:
    """FitAnalyzerAgent.act() must return a string even with no JD/resume."""
    import json
    from tests.conftest import JsonLLM
    from agent_test.agents.fit_analyzer.agent import FitAnalyzerAgent

    payload = json.dumps({
        "has_jd": False, "has_resume": False,
        "clarification_needed": True,
        "clarification_question": "Paste JD and resume please.",
        "job_description": "", "resume_text": "",
    })
    agent = FitAnalyzerAgent(llm=JsonLLM(json_response=payload))
    result = agent.act("hi")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------


def test_package_level_exports() -> None:
    """All public symbols must be importable from the top-level agents package."""
    from agent_test.agents import (
        CrewAIAgent,
        DEFAULT_MODEL,
        FitAnalyzerState,
        FitAnalyzerAgent,
        build_fit_analyzer_graph,
        run_fit_analyzer_crew,
    )
    for sym in (CrewAIAgent, DEFAULT_MODEL, FitAnalyzerState,
                FitAnalyzerAgent, build_fit_analyzer_graph, run_fit_analyzer_crew):
        assert sym is not None
