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
from repo_historian.nodes._helpers import build_llm, get_github_token
from repo_historian.state import DiffAnalysis, DiffAnalysisInput


class _AnalysisOutput(BaseModel):
    era_hint: str
    summary: str
    narrative_paragraph: str
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

    llm = build_llm()
    structured_llm = llm.with_structured_output(_AnalysisOutput)

    result: _AnalysisOutput = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software historian analyzing a range of changes between two "
                    "commits in a repository. This range represents a cohesive chapter of "
                    "the project's story. Provide: era_hint (a short thematic label for this "
                    "period of the project), summary (1-2 sentences), narrative_paragraph "
                    "(3-5 sentences telling the story of this arc of change in past tense), "
                    "and key_changes (3-5 bullet points)."
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
    )

    pair_key = f"{state['from_sha']}..{state['to_sha']}"
    all_tags = [t for c in commits_in_range for t in c.tags]
    all_authors = list({c.author for c in commits_in_range})

    analysis = DiffAnalysis(
        pair_key=pair_key,
        from_sha=state["from_sha"],
        to_sha=state["to_sha"],
        era_hint=result.era_hint,
        summary=result.summary,
        narrative_paragraph=result.narrative_paragraph,
        key_changes=result.key_changes,
        additions=additions,
        deletions=deletions,
        start_date=state["from_commit"].date,
        end_date=state["to_commit"].date,
        label=state.get("label", ""),
        commit_count=len(commits_in_range),
        from_message=state["from_commit"].message,
        to_message=state["to_commit"].message,
        authors=all_authors,
        tags=all_tags,
        repo_full_name=state["repo_full_name"],
    )
    return {"diff_analyses": [analysis]}
