"""LangGraph outer graph for the resume fit analyzer.

Architecture
────────────
  input_collector  ─→  (clarification_needed?) ─→  ask_user  ─→  END
                   └──────────────────────────→  crew_node  ─→  END

* ``input_collector``  — LLM node that checks the conversation for a job
  description and a resume.  Returns a structured assessment of what is
  present, what is missing, and (if both are available) the extracted texts.
* ``ask_user``         — Passes the clarification question back to the user
  without running the analysis pipeline.
* ``crew_node``        — Calls :func:`.resume_crew.run_resume_crew`, which
  runs the four-agent CrewAI sequential pipeline and returns the full report.
"""

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .crew import run_resume_crew
from .state import ResumeState


# ---------------------------------------------------------------------------
# Greeting shown on the very first turn (no inputs yet)
# ---------------------------------------------------------------------------

_GREETING = (
    "Please paste the full job description and your resume or qualifications "
    "summary below, and I'll deliver a no-nonsense fit analysis."
)

# ---------------------------------------------------------------------------
# System prompt for the input_collector LLM call
# ---------------------------------------------------------------------------

_INPUT_COLLECTOR_SYSTEM = """You are an input validation assistant for a job fit analyzer.

Your only job is to inspect a conversation and decide whether it contains BOTH:
  (A) A full job description (JD)
  (B) A resume OR a detailed qualifications summary

Rules:
• A greeting, short question, or URL alone does NOT count as a JD or resume.
• If the user provides a URL instead of pasting text, ask them to paste the full text.
• If critical information is vague or ambiguous (e.g. completely undated experience,
  no skills mentioned at all, or a one-line "I have 5 years of Python"), ask ONE
  focused clarifying question before proceeding.
• Never make assumptions to fill gaps — treat absence as a gap.
• If both inputs are present and sufficient, extract them verbatim.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation.
Schema:
{
  "has_jd": true | false,
  "has_resume": true | false,
  "clarification_needed": true | false,
  "clarification_question": "<question to ask the user, or empty string>",
  "job_description": "<extracted JD text, or empty string>",
  "resume_text": "<extracted resume text, or empty string>"
}

Decisions:
• clarification_needed = true   → when has_jd=false OR has_resume=false OR input is ambiguous
• clarification_needed = false  → only when BOTH has_jd=true AND has_resume=true AND no ambiguity
• If clarification_needed=true, set clarification_question to a focused, specific question about
  exactly what is still missing (e.g. "I still need the job description — please paste it.").
  Do NOT repeat a generic welcome message.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conversation_to_str(messages: list[dict[str, str]]) -> str:
    """Format message history as a readable string for the LLM."""
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    """Attempt to parse JSON from an LLM response, stripping markdown fences."""
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: treat as unrecoverable, ask for clarification
    return {
        "has_jd": False,
        "has_resume": False,
        "clarification_needed": True,
        "clarification_question": (
            "I couldn't parse your input. Please paste the full job description "
            "and your resume or qualifications summary as plain text."
        ),
        "job_description": "",
        "resume_text": "",
    }


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _make_input_collector_node(llm: BaseChatModel):
    """Return the ``input_collector`` node function bound to *llm*."""

    def input_collector(state: ResumeState) -> dict:
        """Inspect the conversation and extract JD + resume, or ask for more."""
        messages = state["messages"]

        # Short-circuit for the very first user message: the greeting is returned
        # directly from Python so the LLM is not called yet and cannot repeat it
        # on subsequent turns.
        has_assistant_turn = any(m.get("role") == "assistant" for m in messages)
        if not has_assistant_turn:
            return {
                "clarification_needed": True,
                "clarification_question": _GREETING,
                "job_description": "",
                "resume_text": "",
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

        clarification_needed: bool = parsed.get("clarification_needed", True)
        clarification_question: str = parsed.get("clarification_question", _GREETING)
        job_description: str = parsed.get("job_description", "")
        resume_text: str = parsed.get("resume_text", "")

        # Safety guard: if extraction says both present but texts are empty, ask.
        if not clarification_needed and (not job_description.strip() or not resume_text.strip()):
            clarification_needed = True
            clarification_question = (
                "Please paste the full job description and your resume or "
                "qualifications summary so I can run the analysis."
            )

        return {
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
            "job_description": job_description,
            "resume_text": resume_text,
        }

    return input_collector


def _make_crew_node(crew_llm, get_task_callback=None):
    """Return the ``crew_node`` function that calls the CrewAI pipeline."""

    def crew_node(state: ResumeState) -> dict:
        """Run the four-agent CrewAI pipeline and store the report."""
        cb = get_task_callback() if get_task_callback is not None else None
        report = run_resume_crew(
            llm=crew_llm,
            job_description=state["job_description"],
            resume_text=state["resume_text"],
            task_callback=cb,
        )
        return {
            "analysis_result": report,
            "response": report,
        }

    return crew_node


def _ask_user_node(state: ResumeState) -> dict:
    """Return the clarification question as the response for this turn."""
    return {"response": state["clarification_question"]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_input(state: ResumeState) -> str:
    """Determine next node after input_collector."""
    if state["clarification_needed"]:
        return "ask_user"
    return "crew_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_resume_graph(llm: BaseChatModel, crew_llm=None, get_task_callback=None):
    """Compile and return the resume fit analyzer LangGraph.

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
        :class:`ResumeAgent` so each live analysis can stream step events.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready to be invoked.
    """
    graph: StateGraph = StateGraph(ResumeState)

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
