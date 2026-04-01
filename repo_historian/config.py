"""Central configuration — every magic number lives here."""

VERSION: str = "0.1.0"

# Triage — overlapping-window parameters
TRIAGE_WINDOW_SIZE: int = 500
TRIAGE_MARGIN: int = 50
MIN_INFLECTION_POINTS: int = 3
MAX_INFLECTION_POINTS: int = 12

# Analyze diff — caps
MAX_FILES_PER_DIFF: int = 15
MAX_PATCH_CHARS_PER_FILE: int = 2500
MAX_DIFF_CHARS_TOTAL: int = 14000

# Narrative synthesis
MIN_NARRATIVE_WORDS: int = 1200
MAX_NARRATIVE_WORDS: int = 2500

# LLM — uncomment the model you want to use (one at a time)
# Anthropic
# MODEL_NAME: str = "claude-sonnet-4-6"
# MODEL_NAME: str = "claude-opus-4-6"
# MODEL_NAME: str = "claude-haiku-4-5"
# OpenAI
# MODEL_NAME: str = "gpt-5.4"
MODEL_NAME: str = "gpt-5.4-mini"
# MODEL_NAME: str = "gpt-5.4-nano"
# Gemini
# MODEL_NAME: str = "gemini-3.1-pro-preview"
# MODEL_NAME: str = "gemini-3.1-flash-lite-preview"

LLM_TEMPERATURE: float = 0
MAX_COMPLETION_TOKENS: int = 32_768
REASONING_EFFORT: str = "low"

# Narrative model — separate from the data-gathering model above
# NARRATIVE_MODEL_NAME: str = "claude-opus-4-6"
NARRATIVE_MODEL_NAME: str = "gpt-5.4"
NARRATIVE_LLM_TEMPERATURE: float = 0.7
NARRATIVE_MAX_COMPLETION_TOKENS: int = 16_384


PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def detect_provider(model_name: str) -> str:
    prefixes = [("claude", "anthropic"), ("gemini", "google"), ("gpt", "openai"), ("o", "openai")]
    for prefix, provider in prefixes:
        if model_name.startswith(prefix):
            return provider
    raise ValueError(
        f"Unknown model provider for '{model_name}'. "
        f"Model name must start with: claude, gpt, o, or gemini."
    )
