"""LangGraph outer graph for the resume improvement pipeline.

Architecture
────────────
  input_collector  ─→  (clarification_needed?) ─→  ask_user  ─→  END
                   └──────────────────────────→  crew_node  ─→  END

* ``input_collector``  — LLM node that checks the conversation for:
  (1) the candidate's current resume, (2) a Job Fit Analysis report, and
  (3) the original job description (acceptable if embedded in the analysis).
  Returns a focused clarification question when any required input is missing.
* ``ask_user``         — Passes the clarification question back to the user
  without running the enhancement pipeline.
* ``crew_node``        — Calls :func:`.crew.run_resume_improve_crew`, which
  runs the four-agent CrewAI sequential pipeline and returns the enhancement
  report.
"""

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .crew import run_resume_improve_crew
from .state import ResumeImproveState


# ---------------------------------------------------------------------------
# Greeting shown on the very first turn
# ---------------------------------------------------------------------------

_GREETING = (
    "Please provide the following to get started:\n"
    "  1. Your current resume (paste full text)\n"
    "  2. Your Job Fit Analysis output (from the Resume Fit Analyzer)\n"
    "  3. The original job description (if not included in the analysis)\n\n"
    "I'll use all three to engineer a targeted, ATS-optimized resume built "
    "specifically for this role."
)

# ---------------------------------------------------------------------------
# System prompt for the input_collector LLM call
# ---------------------------------------------------------------------------

_INPUT_COLLECTOR_SYSTEM = """You are an input validation assistant for a resume improvement agent.

Your only job is to inspect a conversation and decide whether it contains ALL of:
  (A) A current resume — the candidate's full resume text (not a URL, not a summary)
  (B) A Job Fit Analysis report — structured output from a fit analyzer (contains sections
      like "JOB FIT ANALYSIS", a fit score, strengths, gaps, ATS keyword analysis, etc.)
  (C) A job description — either pasted separately OR already embedded inside the
      Job Fit Analysis report (it is acceptable to extract the JD from the analysis)

Rules:
• A greeting, short question, or URL alone does NOT count as a resume.
• A brief qualifications summary without work history does NOT count as a full resume.
• The Job Fit Analysis MUST be a structured report, not just a casual mention of one.
• The job description can be extracted from within the Job Fit Analysis if it was the
  basis of that analysis — in this case set has_job_description=true and extract it.
• If any required input is missing, ask ONE focused question about exactly what is missing.
• Never make assumptions to fill gaps — treat absence as a gap.
• If all required inputs are present and sufficient, extract them verbatim.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation.
Schema:
{
  "has_resume": true | false,
  "has_fit_analysis": true | false,
  "has_job_description": true | false,
  "clarification_needed": true | false,
  "clarification_question": "<focused question about what is still missing, or empty string>",
  "resume_text": "<extracted resume text, or empty string>",
  "fit_analysis": "<extracted Job Fit Analysis report, or empty string>",
  "job_description": "<extracted job description text, or empty string>"
}

Decisions:
• clarification_needed = true   → when has_resume=false OR has_fit_analysis=false
• clarification_needed = false  → only when BOTH has_resume=true AND has_fit_analysis=true
• has_job_description=false is acceptable — the pipeline can work with the JD embedded
  in the analysis; do not ask for it if it can be inferred from the fit analysis
• If clarification_needed=true, set clarification_question to a focused, specific question
  about exactly what is still missing (e.g. "I still need your current resume — please
  paste the full text.")
• Do NOT repeat the generic welcome message as a clarification question
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_ASSISTANT_CHARS = 400  # cap assistant msgs to avoid re-sending large HTML reports


def _conversation_to_str(messages: list[dict[str, str]]) -> str:
    """Format message history as a readable string for the LLM.

    Assistant messages longer than ``_MAX_ASSISTANT_CHARS`` are truncated so
    that large HTML enhancement reports from previous turns do not bloat the
    context sent to the input_collector on every subsequent turn.
    """
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        content = m["content"]
        if m["role"] == "assistant" and len(content) > _MAX_ASSISTANT_CHARS:
            content = content[:_MAX_ASSISTANT_CHARS] + "… [truncated]"
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    """Attempt to parse JSON from an LLM response, stripping markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "has_resume": False,
        "has_fit_analysis": False,
        "has_job_description": False,
        "clarification_needed": True,
        "clarification_question": (
            "I couldn't parse your input. Please paste:\n"
            "  1. Your current resume (full text)\n"
            "  2. Your Job Fit Analysis output\n"
            "  3. The original job description (if not in the analysis)"
        ),
        "resume_text": "",
        "fit_analysis": "",
        "job_description": "",
    }


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _make_input_collector_node(llm: BaseChatModel):
    """Return the ``input_collector`` node function bound to *llm*."""

    def input_collector(state: ResumeImproveState) -> dict:
        """Inspect the conversation and extract all required inputs, or ask for more."""
        messages = state["messages"]

        # Short-circuit on the very first user message — greet without an LLM call.
        has_assistant_turn = any(m.get("role") == "assistant" for m in messages)
        if not has_assistant_turn:
            return {
                "clarification_needed": True,
                "clarification_question": _GREETING,
                "resume_text": "",
                "fit_analysis": "",
                "job_description": "",
            }

        conversation = _conversation_to_str(messages)

        lc_messages = [
            SystemMessage(content=_INPUT_COLLECTOR_SYSTEM),
            HumanMessage(content=f"Conversation so far:\n\n{conversation}"),
        ]

        raw = llm.invoke(lc_messages)
        content = raw.content
        if isinstance(content, list):
            content = "".join(str(c) for c in content)
        else:
            content = str(content)

        parsed = _extract_json(content)

        cn = parsed.get("clarification_needed", True)
        if isinstance(cn, str):
            clarification_needed: bool = cn.strip().lower() not in ("false", "0", "no")
        else:
            clarification_needed = bool(cn)

        clarification_question: str = str(parsed.get("clarification_question") or "")
        resume_text: str = str(parsed.get("resume_text") or "")
        fit_analysis: str = str(parsed.get("fit_analysis") or "")
        job_description: str = str(parsed.get("job_description") or "")

        # Safety guards: reject if claimed present but empty.
        if not clarification_needed and not resume_text.strip():
            clarification_needed = True
            clarification_question = (
                "I still need your current resume — please paste the full text."
            )

        if not clarification_needed and not fit_analysis.strip():
            clarification_needed = True
            clarification_question = (
                "I still need your Job Fit Analysis output — please paste it from "
                "the Resume Fit Analyzer."
            )

        if clarification_needed and not clarification_question.strip():
            clarification_question = _GREETING

        return {
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
            "resume_text": resume_text,
            "fit_analysis": fit_analysis,
            "job_description": job_description,
        }

    return input_collector


def _make_crew_node(crew_llm, get_task_callback=None):
    """Return the ``crew_node`` function that calls the CrewAI pipeline."""

    def crew_node(state: ResumeImproveState) -> dict:
        """Run the four-agent CrewAI pipeline and store the enhancement report."""
        cb = get_task_callback() if get_task_callback is not None else None
        report = run_resume_improve_crew(
            llm=crew_llm,
            resume_text=state["resume_text"],
            fit_analysis=state["fit_analysis"],
            job_description=state["job_description"],
            task_callback=cb,
        )
        return {
            "enhancement_result": report,
            "response": report,
        }

    return crew_node


def _ask_user_node(state: ResumeImproveState) -> dict:
    """Return the clarification question as the response for this turn."""
    return {"response": state["clarification_question"]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_input(state: ResumeImproveState) -> str:
    """Determine next node after input_collector."""
    if state["clarification_needed"]:
        return "ask_user"
    return "crew_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_resume_improve_graph(llm: BaseChatModel, crew_llm=None, get_task_callback=None):
    """Compile and return the resume improvement LangGraph.

    Parameters
    ----------
    llm:
        A LangChain ``BaseChatModel`` used by the input_collector LLM call.
    crew_llm:
        A ``crewai.LLM`` (or compatible object) used by the CrewAI pipeline.
        When *None*, falls back to *llm* (useful in tests where the crew is
        fully mocked).
    get_task_callback:
        A zero-argument callable that returns the current per-request
        CrewAI task-completion callback (or ``None``).  Injected by
        :class:`ResumeImproveAgent` so each live run can stream step events.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready to be invoked.
    """
    graph: StateGraph = StateGraph(ResumeImproveState)

    graph.add_node("input_collector", _make_input_collector_node(llm))
    graph.add_node("ask_user", _ask_user_node)
    graph.add_node(
        "crew_node",
        _make_crew_node(
            crew_llm if crew_llm is not None else llm,
            get_task_callback=get_task_callback,
        ),
    )

    graph.set_entry_point("input_collector")

    graph.add_conditional_edges(
        "input_collector",
        _route_after_input,
        {
            "ask_user": "ask_user",
            "crew_node": "crew_node",
        },
    )

    graph.add_edge("ask_user", END)
    graph.add_edge("crew_node", END)

    return graph.compile()
