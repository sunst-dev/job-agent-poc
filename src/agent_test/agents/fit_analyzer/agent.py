"""FitAnalyzerAgent: brutally honest job fit analyzer powered by LangGraph + CrewAI.

Architecture recap
──────────────────
LangGraph (outer graph)
  ├── input_collector  — LLM gate: checks for JD + resume, extracts or asks
  ├── ask_user         — returns clarification question to the user
  └── crew_node        — delegates to CrewAI (4 sequential specialist agents)
          └── CrewAI Crew
                ├── JD Parser Agent
                ├── Resume Analyzer Agent
                ├── Scorer Agent
                └── Report Generator Agent

Usage
─────
    from agent_test.agents import FitAnalyzerAgent

    agent = FitAnalyzerAgent()
    # Turn 1 — no inputs yet
    print(agent.act("hi"))
    # → "Please paste the full job description and your resume …"

    # Turn 2 — user pastes JD + resume
    reply = agent.act(jd_and_resume_text, history=history)
    # → Full structured 🎯 JOB FIT ANALYSIS report
"""

from __future__ import annotations

import queue as _queue
import threading
import time
from typing import Generator

from langchain_core.language_models import BaseChatModel

from agent_test.utils.logger import setup_logger
from agent_test.utils.openrouter_client import get_chat_model, get_crew_llm
from ..base import Agent
from .graph import _GREETING, build_fit_analyzer_graph
from .state import FitAnalyzerState

# Use a capable model for multi-step reasoning; can be overridden at init.
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# Human-readable labels for each LangGraph node.
_STEP_LABELS: dict[str, tuple[str, str]] = {
    "input_collector": ("🔍", "Analyzing input"),
    "ask_user":        ("💬", "Preparing reply"),
    "crew_node":       ("⚙️",  "Running pipeline"),
}

# Labels emitted (in order) for the four sequential CrewAI tasks inside crew_node.
_CREW_STEPS: list[tuple[str, str]] = [
    ("📋", "Parsing job description"),
    ("📄", "Analyzing resume"),
    ("📊", "Scoring fit"),
    ("📝", "Generating report"),
]

_LOGGER = setup_logger(__name__)


def _format_elapsed(seconds: float) -> str:
    """Return a compact human-readable elapsed time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {int(remainder)}s"


def _node_detail(node_name: str, node_output: dict) -> str:
    """Return a human-readable detail string for a completed LangGraph node."""
    if node_name == "input_collector":
        if node_output.get("clarification_needed"):
            q = node_output.get("clarification_question", "").strip()
            return f"Clarification needed\n\n{q}" if q else "Clarification needed"
        jd = node_output.get("job_description", "") or ""
        res = node_output.get("resume_text", "") or ""
        return (
            f"Job description extracted ({len(jd):,} chars)\n"
            f"Resume extracted ({len(res):,} chars)"
        )
    if node_name == "ask_user":
        return (node_output.get("response", "") or "").strip()
    return ""


class FitAnalyzerAgent(Agent):
    """Job fit analyzer that combines LangGraph routing with a CrewAI pipeline.

    On each call to :meth:`act` the agent:

    1. Runs ``input_collector`` — an LLM node that checks the full conversation
       for a job description and a resume.  If either is missing or ambiguous,
       it returns a targeted clarification question.
    2. If both inputs are present, it hands them to a four-agent CrewAI Crew
       (JD Parser → Resume Analyzer → Scorer → Report Generator) that produces
       the full structured fit assessment.

    Parameters
    ----------
    llm:
        A LangChain ``BaseChatModel`` used by the LangGraph input_collector node.
        When *None*, a ``ChatOpenRouter`` instance is created via
        :func:`agent_test.utils.openrouter_client.get_chat_model`.
        A ``crewai.LLM`` is created separately for the CrewAI pipeline.
        Inject a LangChain stub here in tests (the crew is mocked in tests).
    model:
        OpenRouter model identifier (used when *llm* is not provided).
    temperature:
        Sampling temperature (0–2).  Lower values produce more consistent
        structured output.  Defaults to 0.1.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
    ) -> None:
        if llm is None:
            llm = get_chat_model(model=model, temperature=temperature)
            crew_llm = get_crew_llm(model=model, temperature=temperature)
        else:
            # Injected stub (tests) — pass the same object; crew is mocked anyway.
            crew_llm = llm
        self.model = model
        self.temperature = temperature
        self._task_callback = None
        self._graph = build_fit_analyzer_graph(
            llm,
            crew_llm=crew_llm,
            get_task_callback=lambda: self._task_callback,
        )

    def act(
        self,
        observation: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Run one turn and return the reply (convenience wrapper over act_stream)."""
        response = ""
        for event in self.act_stream(observation, history=history):
            if event["type"] == "error":
                raise RuntimeError(str(event["text"]))
            if event["type"] == "response":
                response = event["text"]
        if not response:
            raise RuntimeError("FitAnalyzerAgent produced no response.")
        return response

    def act_stream(
        self,
        observation: str,
        history: list[dict[str, str]] | None = None,
    ) -> Generator[dict, None, None]:
        """Stream graph events for one turn, yielding dicts as they are produced.

        Each yielded dict has a ``"type"`` key:

                * ``{"type": "status", "text": str}``            — transient progress
                    text for the UI while long-running steps are in flight.
        * ``{"type": "step", "icon": str, "label": str}``  — one per node /
          crew task completed; arrives as it finishes (real time).
        * ``{"type": "response", "text": str}``             — final answer,
          yielded after all steps.
        * ``{"type": "error", "text": str}``                — yielded instead
          of ``response`` when the graph raises an exception.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": observation})

        # First-turn shortcircuit — no LLM call needed.
        if not any(m.get("role") == "assistant" for m in messages):
            yield {"type": "response", "text": _GREETING}
            return

        turn_started = time.monotonic()

        initial_state: FitAnalyzerState = {
            "messages": messages,
            "job_description": "",
            "resume_text": "",
            "clarification_needed": False,
            "clarification_question": "",
            "analysis_result": "",
            "response": "",
        }

        event_q: _queue.Queue = _queue.Queue()
        crew_task_iter = iter(_CREW_STEPS)

        yield {
            "type": "status",
            "text": "Starting resume analysis. First I’ll validate the job description and resume.",
        }

        def _on_crew_task(_output) -> None:
            """Called by CrewAI after each sequential task completes."""
            try:
                icon, label = next(crew_task_iter)
                detail = str(getattr(_output, "raw", _output) or "")[:1000]
                elapsed = _format_elapsed(time.monotonic() - turn_started)
                _LOGGER.info("Resume pipeline task completed: %s (%s)", label, elapsed)
                event_q.put({"type": "step", "icon": icon, "label": label, "detail": detail})
            except StopIteration:
                pass

        def _graph_worker() -> None:
            """Run the compiled LangGraph graph in a background thread."""
            self._task_callback = _on_crew_task
            try:
                _LOGGER.info("Resume analysis started")
                final = ""
                for event in self._graph.stream(initial_state):
                    node_name = next(iter(event))
                    node_output = event[node_name]
                    if "response" in node_output:
                        final = str(node_output["response"])
                    # crew_node sub-steps stream via _on_crew_task callback;
                    # emit a plain node label only for non-crew nodes.
                    if node_name != "crew_node":
                        icon, label = _STEP_LABELS.get(node_name, ("🔹", node_name))
                        detail = _node_detail(node_name, node_output)
                        elapsed = _format_elapsed(time.monotonic() - turn_started)
                        _LOGGER.info("Resume graph node completed: %s (%s)", node_name, elapsed)
                        event_q.put({"type": "step", "icon": icon, "label": label, "detail": detail})
                event_q.put({"type": "response", "text": final})
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("Resume analysis failed")
                event_q.put({"type": "error", "text": str(exc)})
            finally:
                self._task_callback = None
                event_q.put(None)  # sentinel

        worker = threading.Thread(target=_graph_worker, daemon=True)
        worker.start()

        _HEARTBEAT_INTERVAL = 15   # seconds between keep-alive pulses
        _TOTAL_TIMEOUT = 1200      # 20-minute hard cutoff (4 tasks × free-tier models)
        deadline = time.monotonic() + _TOTAL_TIMEOUT
        timed_out = False

        while True:
            try:
                item = event_q.get(timeout=_HEARTBEAT_INTERVAL)
            except _queue.Empty:
                if time.monotonic() > deadline:
                    _LOGGER.warning("Resume analysis timed out after %s", _format_elapsed(_TOTAL_TIMEOUT))
                    yield {"type": "error", "text": "Request timed out after 20 minutes."}
                    timed_out = True
                    break
                elapsed = _format_elapsed(time.monotonic() - turn_started)
                yield {
                    "type": "status",
                    "text": f"Still working... elapsed {elapsed}. Free/shared models can pause between steps.",
                }
                continue
            if item is None:
                break
            yield item

        if timed_out and worker.is_alive():
            _LOGGER.warning("Resume worker is still running in the background after timeout")
            return

        worker.join(timeout=0.1)
        _LOGGER.info("Resume analysis finished in %s", _format_elapsed(time.monotonic() - turn_started))
