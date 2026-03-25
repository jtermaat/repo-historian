"""Node: cluster_into_eras."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from repo_historian import logger
from repo_historian.config import MAX_ERAS, MIN_ERAS
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import BatchSummary, Era, GraphState

# --- Structured output schemas (one for each path) ---


class _EraFromBatches(BaseModel):
    title: str
    start_date: str
    end_date: str
    description: str
    batch_indices: list[int]


class _EraFromAnalyses(BaseModel):
    title: str
    start_date: str
    end_date: str
    description: str
    diff_pair_keys: list[str]


class _ErasFromBatchesResponse(BaseModel):
    eras: list[_EraFromBatches] = Field(min_length=MIN_ERAS, max_length=MAX_ERAS)


class _ErasFromAnalysesResponse(BaseModel):
    eras: list[_EraFromAnalyses] = Field(min_length=MIN_ERAS, max_length=MAX_ERAS)


def _cluster_from_batches(
    batch_summaries: list[BatchSummary],
) -> list[Era]:
    """Cluster using batch summaries (large repo path)."""
    logger.info(
        "Grouping %d batch summaries into thematic eras",
        len(batch_summaries),
    )

    summaries_text = "\n\n".join(
        f"BATCH {bs.batch_index} ({bs.start_date} to {bs.end_date}):\n"
        f"Themes: {bs.themes}\n"
        f"Covers {len(bs.pair_keys)} diff pairs."
        for bs in batch_summaries
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_ErasFromBatchesResponse)

    result = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are a software historian. Group the following "
                    f"batch summaries into {MIN_ERAS}-{MAX_ERAS} thematic "
                    f"eras that tell the story of the project's evolution. "
                    f"Each era needs a title, start/end dates, description, "
                    f"and the list of batch indices belonging to it. "
                    f"Every batch must belong to exactly one era."
                )
            ),
            HumanMessage(content=f"Batch summaries:\n{summaries_text}"),
        ]
    )

    batch_by_index = {bs.batch_index: bs for bs in batch_summaries}

    eras: list[Era] = []
    for e in result.eras:
        all_pair_keys: list[str] = []
        for idx in e.batch_indices:
            bs = batch_by_index.get(idx)
            if bs:
                all_pair_keys.extend(bs.pair_keys)
        eras.append(
            Era(
                title=e.title,
                start_date=e.start_date,
                end_date=e.end_date,
                description=e.description,
                diff_pair_keys=all_pair_keys,
            )
        )
    return eras


def _cluster_from_analyses(state: GraphState) -> list[Era]:
    """Cluster raw analyses directly (small repo path)."""
    analyses = state["diff_analyses"]
    logger.info("Grouping %d analyses into thematic eras (direct)", len(analyses))
    analyses_sorted = sorted(analyses, key=lambda a: a.start_date)

    summaries = "\n".join(
        f"- {a.from_sha[:8]}..{a.to_sha[:8]} "
        f"({a.start_date} to {a.end_date}): "
        f"era_hint={a.era_hint!r}, summary={a.summary}"
        for a in analyses_sorted
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_ErasFromAnalysesResponse)

    result = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are a software historian. Group the following "
                    f"diff-pair analyses into {MIN_ERAS}-{MAX_ERAS} "
                    f"thematic eras that tell the story of the project's "
                    f"evolution. Each era needs a title, start/end dates, "
                    f"description, and the list of diff pair keys (in "
                    f"from..to format using 8-char SHA prefixes as shown) "
                    f"belonging to it. Every diff pair must belong to "
                    f"exactly one era."
                )
            ),
            HumanMessage(content=f"Diff-pair analyses:\n{summaries}"),
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


def cluster_into_eras(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    batch_summaries = state.get("batch_summaries", [])

    if batch_summaries:
        eras = _cluster_from_batches(batch_summaries)
    else:
        eras = _cluster_from_analyses(state)

    return {"eras": eras}
