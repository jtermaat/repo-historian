"""Node: cross_repo_expand_narrative — ecosystem-level narrative synthesis."""

from __future__ import annotations

from itertools import groupby
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repo_historian import logger
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import (
    BatchSummary,
    DiffAnalysis,
    MultiRepoGraphState,
    RepoMetadata,
)


def _build_cast_of_characters(all_metadata: list[RepoMetadata]) -> str:
    """Build a 'cast of characters' section describing each repo's role."""
    lines: list[str] = []
    for meta in all_metadata:
        lines.append(
            f"- {meta.full_name}: {meta.description} "
            f"(Language: {meta.language}, Stars: {meta.stars})"
        )
    return "\n".join(lines)


def _build_era_blocks_from_batches(state: MultiRepoGraphState) -> str:
    """Build era context from batch summaries (large repo path)."""
    eras = state["cross_repo_eras"]
    batch_summaries = state["merged_batch_summaries"]
    analyses = state["merged_analyses"]
    analyses_by_key: dict[str, DiffAnalysis] = {a.pair_key: a for a in analyses}

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

        # Group analyses by repo for clarity
        era_analyses = [analyses_by_key[pk] for pk in era.diff_pair_keys if pk in analyses_by_key]
        era_analyses.sort(key=lambda a: (a.repo_full_name, a.start_date))

        for repo_name, group in groupby(era_analyses, key=lambda a: a.repo_full_name):
            block_lines.append(f"\n  [{repo_name}]:")
            seen_batches: set[int] = set()
            for a in group:
                bs = pair_key_to_batch.get(a.pair_key)
                if bs and bs.batch_index not in seen_batches:
                    seen_batches.add(bs.batch_index)
                    block_lines.append(
                        f"    Period {bs.start_date} to {bs.end_date}: {bs.narrative_digest}"
                    )
                compare_url = (
                    f"https://github.com/{a.repo_full_name}/compare/{a.from_sha}...{a.to_sha}"
                )
                short = f"{a.from_sha[:8]}..{a.to_sha[:8]}"
                if a.pair_key in [rp for bs2 in batch_summaries for rp in bs2.representative_pairs]:
                    block_lines.append(f"    Citation: [{short}]({compare_url})")

        era_blocks.append("\n".join(block_lines))

    return "\n\n".join(era_blocks)


def _build_era_blocks_from_analyses(state: MultiRepoGraphState) -> str:
    """Build era context from raw analyses (small repos path)."""
    eras = state["cross_repo_eras"]
    analyses = state["merged_analyses"]
    analyses_by_key: dict[str, DiffAnalysis] = {a.pair_key: a for a in analyses}

    era_blocks: list[str] = []
    for era in eras:
        block_lines = [f"ERA: {era.title} ({era.start_date} – {era.end_date})"]
        block_lines.append(f"Description: {era.description}")

        era_analyses = [analyses_by_key[pk] for pk in era.diff_pair_keys if pk in analyses_by_key]
        era_analyses.sort(key=lambda a: (a.repo_full_name, a.start_date))

        for repo_name, group in groupby(era_analyses, key=lambda a: a.repo_full_name):
            block_lines.append(f"\n  [{repo_name}]:")
            for a in group:
                compare_url = (
                    f"https://github.com/{a.repo_full_name}/compare/{a.from_sha}...{a.to_sha}"
                )
                short = f"{a.from_sha[:8]}..{a.to_sha[:8]}"
                block_lines.append(f"    [{short}]({compare_url}): {a.narrative_paragraph}")

        era_blocks.append("\n".join(block_lines))

    return "\n\n".join(era_blocks)


def cross_repo_expand_narrative(
    state: MultiRepoGraphState, config: RunnableConfig
) -> dict[str, Any]:
    logger.info("Generating cross-repo ecosystem narrative")
    all_metadata = state["all_repo_metadata"]
    batch_summaries = state.get("merged_batch_summaries", [])

    cast = _build_cast_of_characters(all_metadata)

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
                    "narrative history of an ECOSYSTEM of related GitHub "
                    "repositories. These projects are interconnected — they "
                    "share maintainers, depend on each other, or serve "
                    "complementary roles.\n\n"
                    "Write in past tense with an authoritative yet engaging "
                    "voice. Structure your narrative around the provided eras. "
                    "Include an introduction that sets the stage for the "
                    "ecosystem and a conclusion reflecting on its evolution.\n\n"
                    "KEY GUIDANCE:\n"
                    "- Weave the stories of the different repositories "
                    "together. When changes in one repo enabled, motivated, "
                    "or coincided with changes in another, make that "
                    "connection explicit.\n"
                    "- Use temporal proximity as a signal: if repo A adds an "
                    "API and repo B starts using it shortly after, that is a "
                    "causal connection worth narrating.\n"
                    "- Give each repository narrative space proportional to "
                    "its significance and activity level.\n"
                    "- Write as much as needed to tell the full story "
                    "thoroughly — be comprehensive but not repetitive.\n"
                    "CRITICAL: Cite changes inline using Markdown links "
                    "in the format "
                    "[[from..to]](github_compare_url) — e.g. "
                    "[[a1b2c3d4..e5f6g7h8]](https://github.com/owner/"
                    "repo/compare/sha1...sha2). "
                    "Include causal analysis — explain WHY changes "
                    "happened and how they connected across repositories. "
                    "Every era must appear as a section."
                )
            ),
            HumanMessage(
                content=(
                    f"Ecosystem repositories:\n{cast}\n\n"
                    f"Era data with diff-pair narratives:\n\n"
                    f"{all_eras_text}"
                )
            ),
        ]
    )

    narrative = response.text

    return {"narrative": narrative}
