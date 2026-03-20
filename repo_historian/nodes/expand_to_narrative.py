"""Node: expand_to_narrative."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repo_historian.config import MAX_NARRATIVE_WORDS, MIN_NARRATIVE_WORDS
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import CommitAnalysis, GraphState


def expand_to_narrative(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    meta = state["repo_metadata"]
    eras = state["eras"]
    analyses_by_sha: dict[str, CommitAnalysis] = {a.sha: a for a in state["commit_analyses"]}

    # Build per-era context for the LLM
    era_blocks: list[str] = []
    for era in eras:
        block_lines = [f"ERA: {era.title} ({era.start_date} – {era.end_date})"]
        block_lines.append(f"Description: {era.description}")
        for sha in era.commit_shas:
            a = analyses_by_sha.get(sha)
            if not a:
                continue
            commit_url = f"https://github.com/{meta.full_name}/commit/{sha}"
            block_lines.append(f"  [{sha[:8]}]({commit_url}): {a.narrative_paragraph}")
        era_blocks.append("\n".join(block_lines))

    all_eras_text = "\n\n".join(era_blocks)

    llm = build_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software historian writing a compelling narrative history of a "
                    "GitHub repository. Write in past tense with an authoritative yet engaging "
                    "voice. Structure your narrative around the provided eras. Include an "
                    "introduction paragraph and a conclusion paragraph. "
                    f"Target length: {MIN_NARRATIVE_WORDS}-{MAX_NARRATIVE_WORDS} words. "
                    "CRITICAL: Cite commits inline using Markdown links in the format "
                    "[[sha_prefix]](github_commit_url) -- e.g. "
                    "[[a1b2c3d4]](https://github.com/owner/repo/commit/full_sha). "
                    "Include causal analysis -- explain WHY changes happened and how they "
                    "connected to each other. Every era must appear as a section."
                )
            ),
            HumanMessage(
                content=(
                    f"Repository: {meta.full_name}\n"
                    f"Description: {meta.description}\n"
                    f"Language: {meta.language} | Stars: {meta.stars}\n\n"
                    f"Era data with commit narratives:\n\n{all_eras_text}"
                )
            ),
        ]
    )

    narrative = response.text

    return {"narrative": narrative}
