"""Node: write_narrative — single LLM call to produce the final narrative."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repo_historian import logger
from repo_historian.config import MAX_NARRATIVE_WORDS, MIN_NARRATIVE_WORDS
from repo_historian.nodes._helpers import build_narrative_llm
from repo_historian.state import DiffAnalysis, GraphState, MultiRepoGraphState, RepoMetadata


def _format_analyses(
    analyses: list[DiffAnalysis],
) -> list[dict[str, Any]]:
    """Serialize analyses to dicts with computed compare URLs."""
    return [
        {
            "label": a.label,
            "summary": a.summary,
            "narrative_paragraph": a.narrative_paragraph,
            "key_changes": a.key_changes,
            "start_date": a.start_date,
            "end_date": a.end_date,
            "commit_count": a.commit_count,
            "additions": a.additions,
            "deletions": a.deletions,
            "tags": a.tags,
            "authors": a.authors,
            "repo_full_name": a.repo_full_name,
            "compare_url": (
                f"https://github.com/{a.repo_full_name}/compare/{a.from_sha}...{a.to_sha}"
            ),
            "short_ref": f"{a.from_sha[:8]}..{a.to_sha[:8]}",
        }
        for a in sorted(analyses, key=lambda a: a.start_date)
    ]


def _build_system_prompt(*, is_multi_repo: bool, style: str | None) -> str:
    subject = (
        "the following ecosystem of related GitHub repositories"
        if is_multi_repo
        else "the repository described below"
    )
    base = (
        "You are a software historian. Write a narrative history of "
        f"{subject}, based on the diff analyses provided as JSON. "
        "Include an introduction and a conclusion. "
        "Mention and explain any interesting technical concepts you encounter. "
        f"Target length: {MIN_NARRATIVE_WORDS}-{MAX_NARRATIVE_WORDS} words. "
        "Cite changes inline using Markdown links in the format "
        "[[from..to]](compare_url)."
    )

    if is_multi_repo:
        base += (
            " Weave the stories of the different repositories together. "
            "When changes in one repo enabled, motivated, or coincided with "
            "changes in another, make that connection explicit."
        )

    if style:
        base += f" Write in the style of {style}."

    return base


def _format_metadata(metadata: list[RepoMetadata]) -> list[dict[str, str | int]]:
    return [
        {
            "full_name": m.full_name,
            "description": m.description,
            "language": m.language,
            "stars": m.stars,
        }
        for m in metadata
    ]


def write_narrative(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Single-repo narrative: one LLM call over all diff analyses."""
    logger.info("Generating narrative (single LLM call)")
    meta = state["repo_metadata"]
    analyses = state["diff_analyses"]

    style = config.get("configurable", {}).get("style")
    system_prompt = _build_system_prompt(is_multi_repo=False, style=style)

    payload = {
        "repository": _format_metadata([meta])[0],
        "diff_analyses": _format_analyses(analyses),
    }

    llm = build_narrative_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, indent=2, default=str)),
        ]
    )

    return {"narrative": response.text}


def write_cross_repo_narrative(
    state: MultiRepoGraphState, config: RunnableConfig
) -> dict[str, Any]:
    """Multi-repo narrative: one LLM call over merged diff analyses."""
    logger.info("Generating cross-repo narrative (single LLM call)")
    all_metadata = state["all_repo_metadata"]
    analyses = state["merged_analyses"]

    style = config.get("configurable", {}).get("style")
    system_prompt = _build_system_prompt(is_multi_repo=True, style=style)

    payload = {
        "repositories": _format_metadata(all_metadata),
        "diff_analyses": _format_analyses(analyses),
    }

    llm = build_narrative_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, indent=2, default=str)),
        ]
    )

    return {"narrative": response.text}
