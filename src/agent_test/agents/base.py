"""Base classes and interfaces for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re


_GREETING_TURN_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|sup|what'?s up|good morning|good afternoon|good evening|greetings)[!.?\s]*$",
    re.IGNORECASE,
)


def should_send_welcome_greeting(
    observation: str,
    history: list[dict[str, str]] | None = None,
) -> bool:
    """Return ``True`` when a new session should answer with its welcome copy.

    The specialized agents should still greet on a simple first-turn hello, but
    they should not discard substantive first-turn inputs such as a full job
    description, resume, or pipeline handoff payload.
    """
    if history:
      return False

    return bool(_GREETING_TURN_RE.fullmatch((observation or "").strip()))


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
