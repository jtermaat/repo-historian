"""Node: expand_to_narrative."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repo_historian import logger
from repo_historian.config import MAX_NARRATIVE_WORDS, MIN_NARRATIVE_WORDS
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import BatchSummary, DiffAnalysis, GraphState


def _build_era_blocks_from_batches(state: GraphState) -> str:
    """Build era context from batch summaries (large repo path)."""
    meta = state["repo_metadata"]
    eras = state["eras"]
    batch_summaries = state["batch_summaries"]
    analyses_by_key: dict[str, DiffAnalysis] = {a.pair_key: a for a in state["diff_analyses"]}

    # Index: pair_key → batch summary
    pair_key_to_batch: dict[str, BatchSummary] = {}
    for bs in batch_summaries:
        for pk in bs.pair_keys:
            pair_key_to_batch[pk] = bs

    era_blocks: list[str] = []
    for era in eras:
        block_lines = [f"ERA: {era.title} ({era.start_date} – {era.end_date})"]
        block_lines.append(f"Description: {era.description}")
        block_lines.append(f"Diff pairs in this era: {len(era.diff_pair_keys)}")

        # Collect unique batch summaries for this era
        seen_batches: set[int] = set()
        era_batches: list[BatchSummary] = []
        for pk in era.diff_pair_keys:
            bs = pair_key_to_batch.get(pk)
            if bs and bs.batch_index not in seen_batches:
                seen_batches.add(bs.batch_index)
                era_batches.append(bs)
        era_batches.sort(key=lambda b: b.batch_index)

        for bs in era_batches:
            block_lines.append(f"\n  Period {bs.start_date} to {bs.end_date}:")
            block_lines.append(f"  {bs.narrative_digest}")
            for rp_key in bs.representative_pairs:
                a = analyses_by_key.get(rp_key)
                if a:
                    compare_url = (
                        f"https://github.com/{meta.full_name}/compare/{a.from_sha}...{a.to_sha}"
                    )
                    short = f"{a.from_sha[:8]}..{a.to_sha[:8]}"
                    block_lines.append(f"  Citation: [{short}]({compare_url})")

        era_blocks.append("\n".join(block_lines))

    return "\n\n".join(era_blocks)


def _build_era_blocks_from_analyses(state: GraphState) -> str:
    """Build era context from raw analyses (small repo path)."""
    meta = state["repo_metadata"]
    eras = state["eras"]
    analyses_by_key: dict[str, DiffAnalysis] = {a.pair_key: a for a in state["diff_analyses"]}

    era_blocks: list[str] = []
    for era in eras:
        block_lines = [f"ERA: {era.title} ({era.start_date} – {era.end_date})"]
        block_lines.append(f"Description: {era.description}")
        for pair_key in era.diff_pair_keys:
            a = analyses_by_key.get(pair_key)
            if not a:
                continue
            compare_url = f"https://github.com/{meta.full_name}/compare/{a.from_sha}...{a.to_sha}"
            short = f"{a.from_sha[:8]}..{a.to_sha[:8]}"
            block_lines.append(f"  [{short}]({compare_url}): {a.narrative_paragraph}")
        era_blocks.append("\n".join(block_lines))

    return "\n\n".join(era_blocks)


def expand_to_narrative(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    logger.info("Generating final narrative")
    meta = state["repo_metadata"]
    batch_summaries = state.get("batch_summaries", [])

    if batch_summaries:
        all_eras_text = _build_era_blocks_from_batches(state)
    else:
        all_eras_text = _build_era_blocks_from_analyses(state)

    llm = build_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software historian writing a compelling "
                    "narrative history of a GitHub repository. Write in "
                    "past tense with an authoritative yet engaging voice. "
                    "Structure your narrative around the provided eras. "
                    "Include an introduction paragraph and a conclusion "
                    "paragraph. "
                    f"Target length: {MIN_NARRATIVE_WORDS}-"
                    f"{MAX_NARRATIVE_WORDS} words. "
                    "CRITICAL: Cite changes inline using Markdown links "
                    "in the format "
                    "[[from..to]](github_compare_url) — e.g. "
                    "[[a1b2c3d4..e5f6g7h8]](https://github.com/owner/"
                    "repo/compare/sha1...sha2). "
                    "Include causal analysis — explain WHY changes "
                    "happened and how they connected to each other. "
                    "Every era must appear as a section."
                )
            ),
            HumanMessage(
                content=(
                    f"Repository: {meta.full_name}\n"
                    f"Description: {meta.description}\n"
                    f"Language: {meta.language} | Stars: {meta.stars}\n\n"
                    f"Era data with diff-pair narratives:\n\n"
                    f"{all_eras_text}"
                )
            ),
        ]
    )

    narrative = response.text

    return {"narrative": narrative}
