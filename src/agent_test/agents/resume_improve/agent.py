"""ResumeImproveAgent: targeted resume rewriter powered by LangGraph + CrewAI.

Architecture recap
──────────────────
LangGraph (outer graph)
  ├── input_collector  — LLM gate: validates resume, Job Fit Analysis, and JD present
  ├── ask_user         — returns clarification question to the user
  └── crew_node        — delegates to CrewAI (4 sequential specialist agents)
          └── CrewAI Crew
                ├── Enhancement Strategist
                ├── Resume Auditor
                ├── Resume Rewriter
                └── Report Generator

Usage
─────
    from agent_test.agents import ResumeImproveAgent

    agent = ResumeImproveAgent()
    reply = agent.act(resume_fit_analysis_and_jd_text, history=history)
    # → Full structured 📋 RESUME ENHANCEMENT REPORT
"""

from __future__ import annotations

import queue as _queue
import threading
import time
from typing import Generator

from langchain_core.language_models import BaseChatModel

from agent_test.utils.logger import setup_logger
from agent_test.utils.openrouter_client import get_chat_model, get_crew_llm
from ..base import Agent, should_send_welcome_greeting
from .graph import _GREETING, build_resume_improve_graph
from .state import ResumeImproveState
from agent_test.config import DEFAULT_MODEL, RESUME_IMPROVE_TEMPERATURE

# Human-readable labels for each LangGraph node.
_STEP_LABELS: dict[str, tuple[str, str]] = {
    "input_collector": ("🔍", "Analyzing input"),
    "ask_user":        ("💬", "Preparing reply"),
    "crew_node":       ("⚙️",  "Running pipeline"),
}

# Labels emitted (in order) for the four sequential CrewAI tasks inside crew_node.
_CREW_STEPS: list[tuple[str, str]] = [
    ("🎯", "Extracting enhancement targets"),
    ("🔍", "Auditing current resume"),
    ("✍️",  "Rewriting and enhancing"),
    ("📋", "Generating report"),
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
        resume = node_output.get("resume_text", "") or ""
        fit = node_output.get("fit_analysis", "") or ""
        return (
            f"Resume extracted ({len(resume):,} chars)\n"
            f"Fit analysis extracted ({len(fit):,} chars)"
        )
    if node_name == "ask_user":
        return (node_output.get("response", "") or "").strip()
    return ""


class ResumeImproveAgent(Agent):
    """Resume improvement agent combining LangGraph routing with a CrewAI pipeline.

    Requires three inputs from the user: (1) their current resume, (2) a Job Fit
    Analysis output from the fit analyzer, and optionally (3) the original job
    description.  Once all required inputs are present, it runs a four-agent CrewAI
    Crew to produce a targeted, ATS-optimized resume enhancement report.

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
        Sampling temperature (0–2).  Defaults to 0.1.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = RESUME_IMPROVE_TEMPERATURE,
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
        self._graph = build_resume_improve_graph(
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
            raise RuntimeError("ResumeImproveAgent produced no response.")
        return response

    def act_stream(
        self,
        observation: str,
        history: list[dict[str, str]] | None = None,
    ) -> Generator[dict, None, None]:
        """Stream graph events for one turn, yielding dicts as they are produced.

        Each yielded dict has a ``"type"`` key:

        * ``{"type": "status", "text": str}``            — transient progress text.
        * ``{"type": "step", "icon": str, "label": str}``  — one per node / task.
        * ``{"type": "response", "text": str}``             — final answer.
        * ``{"type": "error", "text": str}``                — on graph exception.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": observation})

        # Preserve the lightweight greeting for a simple first-turn hello,
        # but process substantive first-turn inputs immediately.
        if should_send_welcome_greeting(observation, history=history):
            yield {"type": "response", "text": _GREETING}
            return

        turn_started = time.monotonic()

        initial_state: ResumeImproveState = {
            "messages": messages,
            "resume_text": "",
            "fit_analysis": "",
            "job_description": "",
            "clarification_needed": False,
            "clarification_question": "",
            "enhancement_result": "",
            "response": "",
        }

        event_q: _queue.Queue = _queue.Queue()
        crew_task_iter = iter(_CREW_STEPS)

        yield {
            "type": "status",
            "text": "Starting resume enhancement. First I'll validate your inputs.",
        }

        def _on_crew_task(_output) -> None:
            """Called by CrewAI after each sequential task completes."""
            try:
                icon, label = next(crew_task_iter)
                detail = str(getattr(_output, "raw", _output) or "")[:1000]
                elapsed = _format_elapsed(time.monotonic() - turn_started)
                _LOGGER.info("Improve pipeline task completed: %s (%s)", label, elapsed)
                event_q.put({"type": "step", "icon": icon, "label": label, "detail": detail})
            except StopIteration:
                pass

        def _graph_worker() -> None:
            """Run the compiled LangGraph graph in a background thread."""
            self._task_callback = _on_crew_task
            try:
                _LOGGER.info("Resume improvement started")
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
                        _LOGGER.info("Improve graph node completed: %s (%s)", node_name, elapsed)
                        event_q.put({"type": "step", "icon": icon, "label": label, "detail": detail})
                event_q.put({"type": "response", "text": final})
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("Resume improvement failed")
                event_q.put({"type": "error", "text": str(exc)})
            finally:
                self._task_callback = None
                event_q.put(None)  # sentinel

        worker = threading.Thread(target=_graph_worker, daemon=True)
        worker.start()

        _HEARTBEAT_INTERVAL = 15   # seconds between keep-alive pulses
        _TOTAL_TIMEOUT = 1200      # 20-minute hard cutoff
        deadline = time.monotonic() + _TOTAL_TIMEOUT
        timed_out = False

        while True:
            try:
                item = event_q.get(timeout=_HEARTBEAT_INTERVAL)
            except _queue.Empty:
                if time.monotonic() > deadline:
                    _LOGGER.warning(
                        "Resume improvement timed out after %s",
                        _format_elapsed(_TOTAL_TIMEOUT),
                    )
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
            _LOGGER.warning("Improve worker is still running in the background after timeout")
            return

        worker.join(timeout=0.1)
        _LOGGER.info(
            "Resume improvement finished in %s",
            _format_elapsed(time.monotonic() - turn_started),
        )
