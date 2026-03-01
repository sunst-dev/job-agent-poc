"""Shared test fixtures and stubs for agent-test."""

from __future__ import annotations

from typing import Any, Optional

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ---------------------------------------------------------------------------
# Reusable LLM stubs
# ---------------------------------------------------------------------------


class FixedLLM(BaseChatModel):
    """Always returns the same fixed string regardless of input."""

    reply: str = "fixed reply"

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))]
        )


class JsonLLM(BaseChatModel):
    """Returns a configurable JSON string — used for input_collector tests."""

    json_response: str

    @property
    def _llm_type(self) -> str:
        return "json"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self.json_response))
            ]
        )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_llm() -> FixedLLM:
    return FixedLLM()
