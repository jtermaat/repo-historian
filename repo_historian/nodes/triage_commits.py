"""Node: triage_commits."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repo_historian.nodes._helpers import build_llm
from repo_historian.state import GraphState, TriageScore


class _TriageItem(BaseModel):
    sha: str
    focus: bool
    reason: str


class _TriageResponse(BaseModel):
    scores: list[_TriageItem]


def triage_commits(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    commits = state["all_commits"]
    triage_cfg = state["triage_config"]
    batch_size = triage_cfg.batch_size

    # Always-include SHAs: first, last, all tagged
    always_include: set[str] = set()
    if commits:
        always_include.add(commits[0].sha)
        always_include.add(commits[-1].sha)
    for c in commits:
        if c.tags:
            always_include.add(c.sha)

    # Triage in batches
    llm = build_llm()
    structured_llm = llm.with_structured_output(_TriageResponse)
    all_scores: list[TriageScore] = []

    for i in range(0, len(commits), batch_size):
        batch = commits[i : i + batch_size]
        commit_lines = "\n".join(f"- {c.sha[:8]} ({c.date}): {c.message}" for c in batch)

        result: _TriageResponse = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a commit triage assistant. Decide whether each commit is "
                        "focus-worthy for a historical narrative. Mark `focus: true` for commits "
                        "that represent significant features, breaking changes, architecture "
                        "shifts, or meaningful milestones. Mark `focus: false` for routine "
                        "maintenance, typo fixes, version bumps, dependency updates, and "
                        "trivial changes."
                    )
                ),
                HumanMessage(content=f"Triage these commits:\n{commit_lines}"),
            ]
        )

        for item in result.scores:
            all_scores.append(TriageScore(sha=item.sha, focus=item.focus, reason=item.reason))

    # Build sha-prefix -> full-sha lookup
    prefix_to_full: dict[str, str] = {}
    for c in commits:
        prefix_to_full[c.sha[:8]] = c.sha

    # Select commits marked as focus-worthy, then merge always_include
    selected: set[str] = set()
    for s in all_scores:
        if s.focus:
            full_sha = prefix_to_full.get(s.sha, s.sha)
            selected.add(full_sha)
    selected |= always_include

    # Filter to only SHAs that actually exist in our commit list
    valid_shas = {c.sha for c in commits}
    selected &= valid_shas

    # Confirm with user before proceeding to expensive analysis
    ordered = sorted(
        selected, key=lambda s: next((i for i, c in enumerate(commits) if c.sha == s), 0)
    )
    print(f"\nTriage selected {len(ordered)} of {len(commits)} commits for deep analysis.")
    answer = input("Proceed? [Y/n] ").strip().lower()
    if answer and answer != "y":
        print("Aborted by user.")
        raise SystemExit(0)

    return {"significant_commit_shas": ordered}
