"""Agent implementations available in the package."""

# Generic CrewAI agent
from .crewai_agent import CrewAIAgent, DEFAULT_MODEL  # noqa: F401

# Resume fit analyzer
from .fit_analyzer.state import ResumeState  # noqa: F401
from .fit_analyzer.crew import run_resume_crew  # noqa: F401
from .fit_analyzer.graph import build_resume_graph  # noqa: F401
from .fit_analyzer.agent import ResumeAgent  # noqa: F401

# Resume improvement (placeholder — full implementation pending)
from .resume_improve.agent import ResumeImproveAgent  # noqa: F401
