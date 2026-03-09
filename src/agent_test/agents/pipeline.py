"""Pipeline orchestrator — runs fit-analyze → improve → re-analyze in sequence.

Usage
─────
    from agent_test.agents.pipeline import PipelineOrchestrator

    orch = PipelineOrchestrator()

    # Phase 1: fit analysis
    for event in orch.stream_fit_analysis(resume_text, job_description):
        print(event)   # {"type": "step"|"result"|"error", ...}

    # Phase 2: resume improvement  (requires fit_analysis from phase 1)
    for event in orch.stream_improvement(resume_text, job_description, fit_analysis):
        print(event)   # result event includes "improved_resume" key

    # Phase 3: re-analysis on improved resume
    for event in orch.stream_fit_analysis(improved_resume, job_description):
        print(event)
"""

from __future__ import annotations

import queue as _queue
import re as _re
import threading
from typing import Generator

from agent_test.config import DEFAULT_MODEL, FIT_ANALYZER_TEMPERATURE, RESUME_IMPROVE_TEMPERATURE
from agent_test.utils.logger import setup_logger
from agent_test.utils.openrouter_client import get_crew_llm
from .fit_analyzer.crew import run_fit_analyzer_crew
from .resume_improve.crew import run_resume_improve_crew

_LOGGER = setup_logger(__name__)

_FIT_STEPS: list[tuple[str, str]] = [
    ("📋", "Parsing job description"),
    ("📄", "Analyzing resume"),
    ("📊", "Scoring fit"),
    ("📝", "Generating report"),
]

_IMPROVE_STEPS: list[tuple[str, str]] = [
    ("🎯", "Extracting enhancement targets"),
    ("🔍", "Auditing current resume"),
    ("✍️",  "Rewriting and enhancing"),
    ("📋", "Generating report"),
]

_IMPROVED_RESUME_RE = _re.compile(
    r'<pre[^>]*class="improve-resume-text"[^>]*>(.*?)</pre>',
    _re.DOTALL,
)


def _extract_improved_resume(html: str) -> str:
    """Pull plain-text improved resume out of the HTML enhancement report."""
    m = _IMPROVED_RESUME_RE.search(html)
    return m.group(1).strip() if m else html


class PipelineOrchestrator:
    """Runs the three-phase pipeline: fit analysis → improvement → re-analysis.

    Parameters
    ----------
    model:
        OpenRouter model identifier used for all pipeline phases.
    fit_temperature:
        Sampling temperature for the fit analysis crew (default 0.1).
    improve_temperature:
        Sampling temperature for the resume improvement crew (default 0.1).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fit_temperature: float = FIT_ANALYZER_TEMPERATURE,
        improve_temperature: float = RESUME_IMPROVE_TEMPERATURE,
    ) -> None:
        self._model = model
        self._fit_temperature = fit_temperature
        self._improve_temperature = improve_temperature

    # ------------------------------------------------------------------
    # Internal streaming helper
    # ------------------------------------------------------------------

    def _stream(
        self,
        target_fn,
        steps: list[tuple[str, str]],
    ) -> Generator[dict, None, None]:
        """Run *target_fn* in a background thread, yielding SSE-style event dicts.

        Each yielded dict has a ``"type"`` key:

        * ``{"type": "step", "icon": str, "label": str}``  — one per crew task
          completed; arrives in real time as each task finishes.
        * ``{"type": "result", "text": str}``               — final text output,
          yielded after all steps.
        * ``{"type": "heartbeat"}``                         — keep-alive tick;
          callers may filter these out before sending to the client.
        * ``{"type": "error", "text": str}``                — yielded instead of
          ``result`` when the crew raises an exception.
        """
        step_iter = iter(steps)
        event_q: _queue.Queue = _queue.Queue()
        result_holder: list = [None]
        error_holder: list = [None]

        def _task_callback(_output) -> None:
            try:
                icon, label = next(step_iter)
                event_q.put({"type": "step", "icon": icon, "label": label})
            except StopIteration:
                pass

        def _worker() -> None:
            try:
                result_holder[0] = target_fn(_task_callback)
            except Exception as exc:
                error_holder[0] = exc
            finally:
                event_q.put(None)  # sentinel signals completion

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            try:
                item = event_q.get(timeout=15.0)
                if item is None:
                    break
                yield item
            except _queue.Empty:
                yield {"type": "heartbeat"}

        if error_holder[0]:
            yield {"type": "error", "text": str(error_holder[0])}
            return

        yield {"type": "result", "text": result_holder[0]}

    # ------------------------------------------------------------------
    # Public phase methods
    # ------------------------------------------------------------------

    def stream_fit_analysis(
        self, resume_text: str, job_description: str
    ) -> Generator[dict, None, None]:
        """Stream fit-analysis events for *resume_text* against *job_description*."""
        llm = get_crew_llm(model=self._model, temperature=self._fit_temperature)
        yield from self._stream(
            lambda cb: run_fit_analyzer_crew(
                llm, job_description, resume_text, task_callback=cb
            ),
            _FIT_STEPS,
        )

    def stream_improvement(
        self,
        resume_text: str,
        job_description: str,
        fit_analysis: str,
    ) -> Generator[dict, None, None]:
        """Stream resume-improvement events.

        The ``result`` event includes an extra ``"improved_resume"`` key
        containing the extracted plain-text improved resume (suitable for
        passing directly to a subsequent :meth:`stream_fit_analysis` call).
        """
        llm = get_crew_llm(model=self._model, temperature=self._improve_temperature)
        for event in self._stream(
            lambda cb: run_resume_improve_crew(
                llm, resume_text, fit_analysis, job_description, task_callback=cb
            ),
            _IMPROVE_STEPS,
        ):
            if event.get("type") == "result":
                event["improved_resume"] = _extract_improved_resume(event["text"])
            yield event
