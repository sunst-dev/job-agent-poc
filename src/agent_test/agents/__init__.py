"""Agent implementations available in the package."""

# Generic CrewAI agent
from .crewai_agent import CrewAIAgent, DEFAULT_MODEL  # noqa: F401

# Resume fit analyzer
from .fit_analyzer.state import FitAnalyzerState  # noqa: F401
from .fit_analyzer.crew import run_fit_analyzer_crew  # noqa: F401
from .fit_analyzer.graph import build_fit_analyzer_graph  # noqa: F401
from .fit_analyzer.agent import FitAnalyzerAgent  # noqa: F401

# Resume improvement (placeholder — full implementation pending)
from .resume_improve.agent import ResumeImproveAgent  # noqa: F401
