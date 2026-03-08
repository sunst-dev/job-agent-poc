"""Base classes and interfaces for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract interface for agents."""

    @abstractmethod
    def act(
        self,
        observation: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Produce a reply based on the current observation and optional history."""
        raise NotImplementedError
