"""Base classes and interfaces for agents."""

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Abstract interface for agents."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Produce an action based on the current observation."""
        raise NotImplementedError
