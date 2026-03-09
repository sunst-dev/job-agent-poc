"""Central model and pipeline configuration.

Edit this file to change the LLM or temperature used by all agents without
touching any agent implementation code.  Individual values can still be
overridden per-instance by passing ``model`` or ``temperature`` directly to
any agent constructor.
"""

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

# OpenRouter model identifier used by all agents when no model is passed at
# construction time.  Browse available models at https://openrouter.ai/models
DEFAULT_MODEL: str = "anthropic/claude-haiku-4.5"

# ---------------------------------------------------------------------------
# Sampling temperatures  (0 = fully deterministic, 2 = max creativity)
# ---------------------------------------------------------------------------

# CrewAIAgent — general-purpose chat assistant
CHAT_TEMPERATURE: float = 0.2

# FitAnalyzerAgent — structured JSON output required; keep low for consistency
FIT_ANALYZER_TEMPERATURE: float = 0.1

# ResumeImproveAgent — structured output; keep low for consistency
RESUME_IMPROVE_TEMPERATURE: float = 0.1
