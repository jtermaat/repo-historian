"""Node: analyze_commit."""

from __future__ import annotations

from typing import Any

from github import Github
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repo_historian.config import (
    MAX_DIFF_CHARS_TOTAL,
    MAX_FILES_PER_COMMIT,
    MAX_PATCH_CHARS_PER_FILE,
)
from repo_historian.nodes._helpers import build_llm, get_github_token
from repo_historian.state import CommitAnalysis, CommitAnalysisInput


class _AnalysisOutput(BaseModel):
    era_hint: str
    summary: str
    narrative_paragraph: str
    key_changes: list[str]


def analyze_commit(state: CommitAnalysisInput, config: RunnableConfig) -> dict[str, Any]:
    """Analyze a single commit -- called via Send() fan-out."""
    g = Github(get_github_token(config))
    repo = g.get_repo(state["repo_full_name"])
    commit = repo.get_commit(state["sha"])

    # Fetch diff with caps
    diff_parts: list[str] = []
    total_chars = 0
    additions = 0
    deletions = 0

    for idx, f in enumerate(commit.files or []):
        if idx >= MAX_FILES_PER_COMMIT:
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

    # Context commits summary
    context_lines = "\n".join(
        f"- {c.sha[:8]} ({c.date}): {c.message}" for c in state["context_commits"]
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_AnalysisOutput)

    result: _AnalysisOutput = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software historian analyzing a single commit. "
                    "Provide: era_hint (a short thematic label for this period of the project), "
                    "summary (1-2 sentences), narrative_paragraph (3-5 sentences telling the "
                    "story of this change in past tense), and key_changes (3-5 bullet points)."
                )
            ),
            HumanMessage(
                content=(
                    f"Commit: {state['sha'][:8]}\n"
                    f"Message: {state['commit_record'].message}\n"
                    f"Author: {state['commit_record'].author}\n"
                    f"Date: {state['commit_record'].date}\n\n"
                    f"Surrounding commits for context:\n{context_lines}\n\n"
                    f"Diff:\n{diff_text}"
                )
            ),
        ]
    )

    analysis = CommitAnalysis(
        sha=state["sha"],
        era_hint=result.era_hint,
        summary=result.summary,
        narrative_paragraph=result.narrative_paragraph,
        key_changes=result.key_changes,
        additions=additions,
        deletions=deletions,
        date=state["commit_record"].date,
        message=state["commit_record"].message,
        author=state["commit_record"].author,
        tags=state["commit_record"].tags,
    )
    return {"commit_analyses": [analysis]}
