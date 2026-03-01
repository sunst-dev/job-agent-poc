"""CrewAI agent: a single-member Crew backed by an OpenRouter LLM."""

from __future__ import annotations

from crewai import Agent as CrewAgent, Crew, Task

from .base import Agent
from agent_test.utils.openrouter_client import get_crew_llm

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


class CrewAIAgent(Agent):
    """Agent powered by a single-member CrewAI Crew.

    Wraps a CrewAI ``Agent`` / ``Task`` / ``Crew`` trio and exposes the
    standard :meth:`act` interface so the UI can swap agent implementations
    interchangeably.

    Parameters
    ----------
    model:
        OpenRouter model identifier.
    temperature:
        Sampling temperature (0–2).
    role:
        The CrewAI agent's role description.
    goal:
        The CrewAI agent's goal statement.
    backstory:
        The CrewAI agent's backstory.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        role: str = "Helpful AI assistant",
        goal: str = "Answer the user's questions accurately and concisely.",
        backstory: str = (
            "You are a knowledgeable assistant with expertise across many domains."
        ),
    ) -> None:
        self._crew_llm = get_crew_llm(model=model, temperature=temperature)
        self.model = model
        self.temperature = temperature
        self._role = role
        self._goal = goal
        self._backstory = backstory

    def act(  # type: ignore[override]
        self,
        observation: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Run a single-task Crew and return the assistant reply.

        Parameters
        ----------
        observation:
            The latest user message.
        history:
            Optional prior turns as ``{"role": ..., "content": ...}`` dicts.
            Prepended to the task description so the model has full context.
        """
        context = ""
        if history:
            context = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in history
            )
            context = f"Conversation so far:\n{context}\n\n"

        crew_agent = CrewAgent(
            role=self._role,
            goal=self._goal,
            backstory=self._backstory,
            llm=self._crew_llm,
            verbose=False,
        )
        task = Task(
            description=f"{context}User: {observation}",
            expected_output="A helpful, accurate reply to the user's message.",
            agent=crew_agent,
        )
        crew = Crew(agents=[crew_agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
