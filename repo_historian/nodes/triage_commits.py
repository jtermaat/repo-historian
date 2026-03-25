"""Node: triage_commits — identify inflection points via overlapping windows."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repo_historian import logger
from repo_historian.config import (
    MAX_INFLECTION_POINTS,
    MIN_INFLECTION_POINTS,
    TRIAGE_MARGIN,
    TRIAGE_WINDOW_SIZE,
)
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import DiffPair, GraphState, InflectionPoint


class _InflectionPointItem(BaseModel):
    sha: str
    label: str


class _TriageResponse(BaseModel):
    inflection_points: list[_InflectionPointItem]


def triage_commits(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    commits = state["all_commits"]
    stride = TRIAGE_WINDOW_SIZE - TRIAGE_MARGIN * 2

    logger.info("Identifying inflection points across %d commits", len(commits))
    if len(commits) <= 1:
        print("Only one commit found; nothing to analyze.")
        return {"diff_pairs": []}

    llm = build_llm()
    structured_llm = llm.with_structured_output(_TriageResponse)
    all_inflection_points: list[InflectionPoint] = []

    num_windows = 0
    for start in range(0, len(commits), stride):
        end = min(start + TRIAGE_WINDOW_SIZE, len(commits))
        window = commits[start:end]
        num_windows += 1

        # First window: no left margin; last window: no right margin.
        # Boundary commits are added implicitly below.
        eval_start = TRIAGE_MARGIN if start > 0 else 0
        eval_end = len(window) - TRIAGE_MARGIN if end < len(commits) else len(window)

        evaluable_shas = {c.sha[:8] for c in window[eval_start:eval_end]}

        commit_lines = "\n".join(f"- {c.sha[:8]} ({c.date}): {c.message}" for c in window)

        eval_first = window[eval_start].sha[:8]
        eval_last = window[eval_end - 1].sha[:8]

        result: _TriageResponse = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a commit triage assistant building a narrative history of a "
                        "software project. Given a chronological list of commits, identify which "
                        "commits are INFLECTION POINTS — moments where the project meaningfully "
                        "changed direction, reached a milestone, or shifted its character.\n\n"
                        "Guidelines:\n"
                        f"- Aim for {MIN_INFLECTION_POINTS}–{MAX_INFLECTION_POINTS} "
                        "inflection points\n"
                        f"- ONLY select commits between {eval_first} and {eval_last} "
                        f"(inclusive) — the rest are context only\n"
                        "- An inflection point marks: a new feature area, architecture shift, "
                        "major refactor, release milestone, or significant change in project "
                        "direction\n"
                        "- The first and last commits of the full history are included "
                        "automatically — do not include them unless they are genuinely "
                        "significant turning points on their own merit\n"
                        "- Skip routine maintenance, typo fixes, dependency bumps\n"
                        "- Use the 8-char SHA prefixes shown\n"
                        "- Label each with a short description of WHY it is a turning point"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Window {num_windows} — identify inflection points "
                        f"(only between {eval_first} and {eval_last}):\n{commit_lines}"
                    )
                ),
            ]
        )

        for item in result.inflection_points:
            if item.sha in evaluable_shas:
                all_inflection_points.append(InflectionPoint(sha=item.sha, label=item.label))

        logger.info(
            "  Window %d: %d commits, %d evaluable, %d inflection points",
            num_windows,
            len(window),
            len(evaluable_shas),
            sum(1 for ip in result.inflection_points if ip.sha in evaluable_shas),
        )

        # If this window reached the end, stop
        if end >= len(commits):
            break

    logger.info("Processed %d windows", num_windows)

    # Resolve SHA prefixes to full SHAs and deduplicate
    prefix_to_full: dict[str, str] = {}
    for c in commits:
        prefix_to_full[c.sha[:8]] = c.sha

    valid_shas = {c.sha for c in commits}
    resolved: list[InflectionPoint] = []
    seen_shas: set[str] = set()
    for ip in all_inflection_points:
        full_sha = prefix_to_full.get(ip.sha, ip.sha)
        if full_sha in valid_shas and full_sha not in seen_shas:
            seen_shas.add(full_sha)
            resolved.append(InflectionPoint(sha=full_sha, label=ip.label))

    # Add implicit boundary inflection points
    first_sha = commits[0].sha
    last_sha = commits[-1].sha
    if first_sha not in seen_shas:
        resolved.append(InflectionPoint(sha=first_sha, label="Project inception"))
    if last_sha not in seen_shas:
        resolved.append(InflectionPoint(sha=last_sha, label="Latest state"))

    # Sort chronologically
    sha_order = {c.sha: i for i, c in enumerate(commits)}
    resolved.sort(key=lambda ip: sha_order.get(ip.sha, 0))

    # Mechanically pair adjacent inflection points
    diff_pairs: list[DiffPair] = []
    for i in range(len(resolved) - 1):
        diff_pairs.append(
            DiffPair(
                from_sha=resolved[i].sha,
                to_sha=resolved[i + 1].sha,
                label=resolved[i + 1].label,
            )
        )

    # Confirm with user before proceeding to expensive analysis
    print(
        f"\nTriage identified {len(resolved)} inflection points, "
        f"forming {len(diff_pairs)} diff ranges:"
    )
    for p in diff_pairs:
        print(f"  {p.from_sha[:8]}..{p.to_sha[:8]}: {p.label}")

    skip_confirmation = config.get("configurable", {}).get("skip_confirmation", False)
    if not skip_confirmation:
        answer = input("Proceed? [Y/n] ").strip().lower()
        if answer and answer != "y":
            print("Aborted by user.")
            raise SystemExit(0)

    return {"diff_pairs": diff_pairs}
