"""Helper for creating language models backed by OpenRouter.

This module loads the API key from the environment and constructs either a
``ChatOpenRouter`` instance (LangChain BaseChatModel) for use in LangGraph
nodes, or a ``crewai.LLM`` instance for use inside CrewAI agents.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


def get_chat_model(
    model: str,
    temperature: float = 0.2,
    *,
    load_env: bool = True,
) -> ChatOpenRouter:
    """Return a configured ``ChatOpenRouter`` instance.

    Reads ``OPENROUTER_API_KEY`` from ``local_test.env`` (by default) and
    constructs the LangChain chat model.  The optional ``load_env`` flag lets
    tests disable file loading to simulate a missing-key scenario.

    Parameters
    ----------
    model:
        OpenRouter model identifier.
    temperature:
        Sampling temperature (0–2).  Lower values reduce hallucination.
    load_env:
        When *True* (default), load ``local_test.env`` before reading the key.

    Raises
    ------
    RuntimeError
        If ``OPENROUTER_API_KEY`` is not set in the environment.
    """
    if load_env:
        load_dotenv("local_test.env")

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    return ChatOpenRouter(model=model, temperature=temperature, api_key=key)


def get_crew_llm(
    model: str,
    temperature: float = 0.2,
    *,
    load_env: bool = True,
):
    """Return a ``crewai.LLM`` instance backed by OpenRouter.

    CrewAI 0.100+ requires its own ``LLM`` wrapper (not a LangChain model)
    when constructing ``Agent`` objects.  This helper creates one using the
    ``openrouter/`` LiteLLM provider prefix.

    Parameters
    ----------
    model:
        OpenRouter model identifier (e.g. ``"arcee-ai/trinity-large-preview:free"``).
    temperature:
        Sampling temperature (0–2).
    load_env:
        When *True* (default), load ``local_test.env`` before reading the key.

    Raises
    ------
    RuntimeError
        If ``OPENROUTER_API_KEY`` is not set in the environment.
    """
    from crewai import LLM  # local import to avoid hard dep at module level

    if load_env:
        load_dotenv("local_test.env")

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    return LLM(
        model=f"openrouter/{model}",
        api_key=key,
        temperature=temperature,
    )
