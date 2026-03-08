"""Placeholder for the Resume Improvement Agent.

TODO: implement the full pipeline.

This agent will accept a resume (and optionally a target role / JD) and
produce concrete, actionable rewrites and suggestions to improve the
resume's impact, clarity, ATS pass-rate, and overall appeal.

Planned architecture (to be finalized once prompts are ready):
──────────────────────────────────────────────────────────────
LangGraph (outer graph)
  ├── input_collector  — validate that a resume (and optional JD) is present
  └── crew_node        — CrewAI sequential pipeline
          ├── Content Analyzer   — identify weak language, vague bullets, gaps
          ├── ATS Optimizer      — keyword alignment, formatting, section order
          ├── Rewriter           — produce concrete rewritten bullet points
          └── Report Generator   — full improvement report with before/after

Placeholder behaviour (current):
  Returns a "Coming soon" message on every turn so the UI button is usable
  and the session management code works end-to-end.
"""

from __future__ import annotations

from ..base import Agent

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

_COMING_SOON = (
    "🚧 Resume Improvement is coming soon!\n\n"
    "This agent will review your resume and deliver concrete rewrites, "
    "ATS optimizations, and a prioritised action plan. "
    "Check back once the prompts are ready."
)


class ResumeImproveAgent(Agent):
    """Placeholder resume improvement agent.

    Returns a 'coming soon' message until the full implementation is ready.
    Replace the body of :meth:`act` (and add :meth:`act_stream`) once the
    prompt design is finalised.
    """

    def act(
        self,
        observation: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return _COMING_SOON
