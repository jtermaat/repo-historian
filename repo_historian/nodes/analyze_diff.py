"""Node: analyze_diff — analyze the diff between two commits."""

from __future__ import annotations

from typing import Any

from github import Github
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repo_historian import logger
from repo_historian.config import (
    MAX_DIFF_CHARS_TOTAL,
    MAX_FILES_PER_DIFF,
    MAX_PATCH_CHARS_PER_FILE,
)
from repo_historian.nodes._helpers import build_fallback_llm, build_llm, get_github_token
from repo_historian.state import DiffAnalysis, DiffAnalysisInput


class _AnalysisOutput(BaseModel):
    key_changes: list[str]


def analyze_diff(state: DiffAnalysisInput, config: RunnableConfig) -> dict[str, Any]:
    """Analyze the diff between two commits — called via Send() fan-out."""
    logger.info(
        "Analyzing %s..%s: %s", state["from_sha"][:8], state["to_sha"][:8], state.get("label", "")
    )
    g = Github(get_github_token(config))
    repo = g.get_repo(state["repo_full_name"])
    comparison = repo.compare(state["from_sha"], state["to_sha"])

    # Fetch diff with caps
    diff_parts: list[str] = []
    total_chars = 0
    additions = 0
    deletions = 0

    for idx, f in enumerate(comparison.files or []):
        if idx >= MAX_FILES_PER_DIFF:
            break
        additions += f.additions
        deletions += f.deletions
        patch = f.patch or ""
        if len(patch) > MAX_PATCH_CHARS_PER_FILE:
            patch = patch[:MAX_PATCH_CHARS_PER_FILE] + "\n... [truncated]"
        if total_chars + len(patch) > MAX_DIFF_CHARS_TOTAL:
            diff_parts.append(f"--- {f.filename} [diff truncated due to total size cap]")
            break
        diff_parts.append(f"--- {f.filename}\n{patch}")
        total_chars += len(patch)

    diff_text = "\n\n".join(diff_parts) if diff_parts else "(no diff available)"

    # Summarize commits in the range for context
    commits_in_range = state["commits_in_range"]
    range_lines = "\n".join(f"- {c.sha[:8]} ({c.date}): {c.message}" for c in commits_in_range)

    messages = [
        SystemMessage(
            content=(
                "You are a software historian analyzing a range of changes between two "
                "commits in a repository. Produce key_changes: bullet points capturing "
                "the most important technical changes, architectural decisions, and "
                "their motivations. Be specific — name concrete technologies, patterns, "
                "config keys, and design tradeoffs. Each bullet should be self-contained "
                "and information-dense. Do not omit anything technically interesting."
            )
        ),
        HumanMessage(
            content=(
                f"Diff range: {state['from_sha'][:8]}..{state['to_sha'][:8]}\n"
                f"From: {state['from_commit'].message} "
                f"({state['from_commit'].date}, {state['from_commit'].author})\n"
                f"To: {state['to_commit'].message} "
                f"({state['to_commit'].date}, {state['to_commit'].author})\n\n"
                f"Commits in this range:\n{range_lines}\n\n"
                f"Diff:\n{diff_text}"
            )
        ),
    ]

    from openai import LengthFinishReasonError

    try:
        llm = build_llm()
        result: _AnalysisOutput = llm.with_structured_output(_AnalysisOutput).invoke(messages)
    except LengthFinishReasonError:
        logger.warning(
            "Primary model hit token limit for %s..%s; retrying with fallback",
            state["from_sha"][:8],
            state["to_sha"][:8],
        )
        fallback = build_fallback_llm()
        result = fallback.with_structured_output(_AnalysisOutput).invoke(messages)

    all_tags = [t for c in commits_in_range for t in c.tags]
    all_authors = list({c.author for c in commits_in_range})

    analysis = DiffAnalysis(
        from_sha=state["from_sha"],
        to_sha=state["to_sha"],
        label=state.get("label", ""),
        key_changes=result.key_changes,
        start_date=state["from_commit"].date,
        end_date=state["to_commit"].date,
        commit_count=len(commits_in_range),
        additions=additions,
        deletions=deletions,
        authors=all_authors,
        tags=all_tags,
        repo_full_name=state["repo_full_name"],
    )
    return {"diff_analyses": [analysis]}
