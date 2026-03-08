"""LangGraph state for the resume improvement pipeline."""

from __future__ import annotations

from typing import TypedDict


class ResumeImproveState(TypedDict):
    """State that flows through the resume improvement graph.

    messages:
        Full conversation history as ``{"role": ..., "content": ...}`` dicts.
    resume_text:
        The candidate's current resume text extracted from the conversation.
    fit_analysis:
        The Job Fit Analysis report produced by the fit analyzer.
    job_description:
        The original job description text (may be extracted from fit_analysis).
    clarification_needed:
        ``True`` when the input collector needs more information from the user.
    clarification_question:
        The single focused question to ask when *clarification_needed* is ``True``.
    enhancement_result:
        The full structured enhancement report produced by the CrewAI pipeline.
    response:
        The message returned to the user for this turn.
    """

    messages: list[dict[str, str]]
    resume_text: str
    fit_analysis: str
    job_description: str
    clarification_needed: bool
    clarification_question: str
    enhancement_result: str
    response: str
