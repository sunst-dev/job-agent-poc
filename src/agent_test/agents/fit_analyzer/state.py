"""LangGraph state for the resume fit analyzer pipeline."""

from __future__ import annotations

from typing import TypedDict


class ResumeState(TypedDict):
    """State that flows through the resume fit analyzer graph.

    messages:
        Full conversation history as ``{"role": ..., "content": ...}`` dicts.
    job_description:
        Extracted job description text parsed from the conversation.
    resume_text:
        Extracted resume / qualifications text parsed from the conversation.
    clarification_needed:
        ``True`` when the input collector needs more information from the user
        before the analysis pipeline can run.
    clarification_question:
        The single focused question to ask the user when
        *clarification_needed* is ``True``.
    analysis_result:
        The full structured report produced by the CrewAI pipeline.
    response:
        The message returned to the user for this turn.
    """

    messages: list[dict[str, str]]
    job_description: str
    resume_text: str
    clarification_needed: bool
    clarification_question: str
    analysis_result: str
    response: str
