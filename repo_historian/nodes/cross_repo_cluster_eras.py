"""Node: cross_repo_cluster_eras — cluster analyses from multiple repos into eras."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from repo_historian import logger
from repo_historian.config import MAX_CROSS_REPO_ERAS, MIN_CROSS_REPO_ERAS
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import (
    BatchSummary,
    DiffAnalysis,
    Era,
    MultiRepoGraphState,
)


class _CrossRepoEra(BaseModel):
    title: str
    start_date: str
    end_date: str
    description: str
    diff_pair_keys: list[str]


class _CrossRepoErasFromBatchesResponse(BaseModel):
    eras: list[_CrossRepoEra] = Field(
        min_length=MIN_CROSS_REPO_ERAS, max_length=MAX_CROSS_REPO_ERAS
    )


class _CrossRepoErasFromAnalysesResponse(BaseModel):
    eras: list[_CrossRepoEra] = Field(
        min_length=MIN_CROSS_REPO_ERAS, max_length=MAX_CROSS_REPO_ERAS
    )


_SYSTEM_PROMPT = (
    "You are a software historian analyzing an ECOSYSTEM of related "
    "repositories. Group the following analyses into thematic eras that "
    "tell the story of how these projects evolved TOGETHER.\n\n"
    "Guidelines:\n"
    f"- Create {MIN_CROSS_REPO_ERAS}-{MAX_CROSS_REPO_ERAS} eras\n"
    "- Eras should capture cross-repo themes: when one project's changes "
    "enabled or motivated changes in another, group them together\n"
    "- Each era needs a title, start/end dates (ISO format), a description, "
    "and the list of diff pair keys belonging to it\n"
    "- Every diff pair must belong to exactly one era\n"
    "- Highlight inter-repo relationships in era descriptions"
)


def _cluster_from_batches(
    batch_summaries: list[BatchSummary],
    analyses: list[DiffAnalysis],
) -> list[Era]:
    """Cluster using batch summaries (large repo path)."""
    logger.info(
        "Cross-repo: grouping %d batch summaries into thematic eras",
        len(batch_summaries),
    )

    summaries_text = "\n\n".join(
        f"BATCH {bs.batch_index} [{bs.repo_full_name}] "
        f"({bs.start_date} to {bs.end_date}):\n"
        f"Themes: {bs.themes}\n"
        f"Covers {len(bs.pair_keys)} diff pairs."
        for bs in batch_summaries
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_CrossRepoErasFromBatchesResponse)

    result = structured_llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(f"Batch summaries from multiple repositories:\n{summaries_text}")
            ),
        ]
    )

    # Resolve diff_pair_keys from the LLM response (8-char SHA format)
    short_to_full: dict[str, str] = {
        f"{a.from_sha[:8]}..{a.to_sha[:8]}": a.pair_key for a in analyses
    }

    eras: list[Era] = []
    for e in result.eras:
        full_keys = [short_to_full.get(k, k) for k in e.diff_pair_keys]
        eras.append(
            Era(
                title=e.title,
                start_date=e.start_date,
                end_date=e.end_date,
                description=e.description,
                diff_pair_keys=full_keys,
            )
        )
    return eras


def _cluster_from_analyses(analyses: list[DiffAnalysis]) -> list[Era]:
    """Cluster raw analyses directly (small repos path)."""
    logger.info(
        "Cross-repo: grouping %d analyses into thematic eras (direct)",
        len(analyses),
    )
    analyses_sorted = sorted(analyses, key=lambda a: a.start_date)

    summaries = "\n".join(
        f"- [{a.repo_full_name}] {a.from_sha[:8]}..{a.to_sha[:8]} "
        f"({a.start_date} to {a.end_date}): "
        f"era_hint={a.era_hint!r}, summary={a.summary}"
        for a in analyses_sorted
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_CrossRepoErasFromAnalysesResponse)

    result = structured_llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(f"Diff-pair analyses from multiple repositories:\n{summaries}")),
        ]
    )

    short_to_full: dict[str, str] = {
        f"{a.from_sha[:8]}..{a.to_sha[:8]}": a.pair_key for a in analyses
    }

    eras: list[Era] = []
    for e in result.eras:
        full_keys = [short_to_full.get(k, k) for k in e.diff_pair_keys]
        eras.append(
            Era(
                title=e.title,
                start_date=e.start_date,
                end_date=e.end_date,
                description=e.description,
                diff_pair_keys=full_keys,
            )
        )
    return eras


def cross_repo_cluster_eras(state: MultiRepoGraphState, config: RunnableConfig) -> dict[str, Any]:
    batch_summaries = state.get("merged_batch_summaries", [])
    analyses = state.get("merged_analyses", [])

    if batch_summaries:
        eras = _cluster_from_batches(batch_summaries, analyses)
    else:
        eras = _cluster_from_analyses(analyses)

    return {"cross_repo_eras": eras}
