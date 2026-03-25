"""Shared helpers for graph nodes."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.types import default_retry_on

from repo_historian.config import (
    LLM_TEMPERATURE,
    MAX_COMPLETION_TOKENS,
    MODEL_NAME,
    REASONING_EFFORT,
    detect_provider,
)


def _retry_on(exc: Exception) -> bool:
    """Don't retry length-limit errors — the same prompt will fail again."""
    from openai import LengthFinishReasonError

    if isinstance(exc, LengthFinishReasonError):
        return False
    return default_retry_on(exc)


def parse_repo_full_name(url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    match = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ValueError(f"Invalid GitHub repo URL: {url}")
    return match.group(1)


def get_github_token(config: RunnableConfig) -> str:
    return config["configurable"]["github_token"]


def build_llm() -> BaseChatModel:
    provider = detect_provider(MODEL_NAME)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort=REASONING_EFFORT,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=MAX_COMPLETION_TOKENS,
        )
    raise ValueError(f"Unsupported provider: {provider}")
