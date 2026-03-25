"""Node: summarize_analyses — compress analyses into batch summaries."""

from __future__ import annotations

import math
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from repo_historian import logger
from repo_historian.config import (
    MAX_ERAS,
    MAX_REPRESENTATIVE_PAIRS,
    MIN_SUMMARIES_PER_ERA,
    SKIP_SUMMARIZATION_FACTOR,
)
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import BatchSummary, GraphState


class _BatchSummaryOutput(BaseModel):
    themes: str
    narrative_digest: str
    representative_pair_keys: list[str] = Field(max_length=MAX_REPRESENTATIVE_PAIRS)


def _compute_batch_size(num_analyses: int, max_eras: int) -> int | None:
    """Compute batch size, or return None to skip summarization."""
    if num_analyses <= max_eras * SKIP_SUMMARIZATION_FACTOR:
        return None
    target = max(max_eras * MIN_SUMMARIES_PER_ERA, int(math.sqrt(num_analyses)))
    return math.ceil(num_analyses / target)


def summarize_analyses(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    analyses = state["diff_analyses"]
    batch_size = _compute_batch_size(len(analyses), MAX_ERAS)

    if batch_size is None:
        logger.info("Only %d analyses — skipping summarization", len(analyses))
        return {"batch_summaries": []}

    analyses_sorted = sorted(analyses, key=lambda a: a.start_date)
    num_batches = math.ceil(len(analyses_sorted) / batch_size)
    logger.info(
        "Summarizing %d analyses in %d batches of ~%d",
        len(analyses_sorted),
        num_batches,
        batch_size,
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_BatchSummaryOutput)
    batch_summaries: list[BatchSummary] = []

    for batch_idx, i in enumerate(range(0, len(analyses_sorted), batch_size)):
        batch = analyses_sorted[i : i + batch_size]

        lines = "\n".join(
            f"- {a.from_sha[:8]}..{a.to_sha[:8]} "
            f"({a.start_date} to {a.end_date}): "
            f"era_hint={a.era_hint!r}, summary={a.summary}"
            for a in batch
        )

        result: _BatchSummaryOutput = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a software historian. Summarize the "
                        "following batch of diff-pair analyses into a "
                        "cohesive thematic overview.\n\n"
                        "Provide:\n"
                        "- themes: 2-3 sentences identifying the major "
                        "themes and directions in this batch.\n"
                        "- narrative_digest: 3-5 sentences of narrative "
                        "prose (past tense) covering the most important "
                        "developments. Include causal connections between "
                        "changes where visible.\n"
                        "- representative_pair_keys: select the 3-5 most "
                        "historically significant diff pair keys (in the "
                        "from..to 8-char SHA format shown) that best "
                        "represent the batch's themes."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Batch {batch_idx + 1} of {num_batches} "
                        f"({batch[0].start_date} to {batch[-1].end_date}):"
                        f"\n{lines}"
                    )
                ),
            ]
        )

        # Resolve short pair keys back to full pair keys
        short_to_full: dict[str, str] = {
            f"{a.from_sha[:8]}..{a.to_sha[:8]}": a.pair_key for a in batch
        }

        batch_summaries.append(
            BatchSummary(
                batch_index=batch_idx,
                start_date=batch[0].start_date,
                end_date=batch[-1].end_date,
                themes=result.themes,
                narrative_digest=result.narrative_digest,
                pair_keys=[a.pair_key for a in batch],
                representative_pairs=[
                    short_to_full.get(k, k) for k in result.representative_pair_keys
                ],
                repo_full_name=batch[0].repo_full_name,
            )
        )
        logger.info(
            "  Batch %d/%d summarized (%d analyses)",
            batch_idx + 1,
            num_batches,
            len(batch),
        )

    return {"batch_summaries": batch_summaries}
