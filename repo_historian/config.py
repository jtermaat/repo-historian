"""Central configuration — every magic number lives here."""

# Triage
DEFAULT_TRIAGE_BATCH_SIZE: int = 80

# Analyze commit — diff caps
MAX_FILES_PER_COMMIT: int = 15
MAX_PATCH_CHARS_PER_FILE: int = 2500
MAX_DIFF_CHARS_TOTAL: int = 14000
CONTEXT_WINDOW_COMMITS: int = 3

# Era clustering
MIN_ERAS: int = 3
MAX_ERAS: int = 7

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
