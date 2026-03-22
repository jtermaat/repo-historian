"""Node: triage_commits — identify diff pairs for narrative analysis."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repo_historian.nodes._helpers import build_llm
from repo_historian.state import DiffPair, GraphState


class _DiffPairItem(BaseModel):
    from_sha: str
    to_sha: str
    label: str


class _TriageResponse(BaseModel):
    pairs: list[_DiffPairItem]


def triage_commits(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    commits = state["all_commits"]
    triage_cfg = state["triage_config"]
    batch_size = triage_cfg.batch_size

    llm = build_llm()
    structured_llm = llm.with_structured_output(_TriageResponse)
    all_pairs: list[DiffPair] = []

    for i in range(0, len(commits), batch_size):
        batch = commits[i : i + batch_size]
        commit_lines = "\n".join(f"- {c.sha[:8]} ({c.date}): {c.message}" for c in batch)

        result: _TriageResponse = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a commit triage assistant building a narrative history of a "
                        "software project. Given a chronological list of commits, identify pairs "
                        "of commits (from_sha, to_sha) whose combined diff tells a cohesive "
                        "chapter of the project's story.\n\n"
                        "Each pair defines a range: from_sha is the starting point and to_sha is "
                        "the endpoint. The diff between them captures everything that changed.\n\n"
                        "Guidelines:\n"
                        "- Identify 5-15 pairs that together cover the most "
                        "important arcs of change\n"
                        "- Pairs should not overlap (no commit should fall "
                        "inside two ranges)\n"
                        "- Pairs may share endpoints (one pair's to_sha can "
                        "be the next pair's from_sha)\n"
                        "- from_sha must come before to_sha chronologically\n"
                        "- Both SHAs must come from the provided list "
                        "(use the 8-char prefixes shown)\n"
                        "- Adjacent commits are fine when a single commit is significant enough\n"
                        "- Skip routine maintenance, typo fixes, dependency bumps — focus on arcs "
                        "that represent features, architecture shifts, migrations, or milestones\n"
                        "- Label each pair with a short description of the change arc"
                    )
                ),
                HumanMessage(
                    content=f"Identify narrative diff pairs from these commits:\n{commit_lines}"
                ),
            ]
        )

        for item in result.pairs:
            all_pairs.append(DiffPair(from_sha=item.from_sha, to_sha=item.to_sha, label=item.label))

    # Build sha-prefix -> full-sha lookup
    prefix_to_full: dict[str, str] = {}
    for c in commits:
        prefix_to_full[c.sha[:8]] = c.sha

    # Resolve prefixes to full SHAs and filter invalid
    valid_shas = {c.sha for c in commits}
    resolved: list[DiffPair] = []
    for p in all_pairs:
        from_full = prefix_to_full.get(p.from_sha, p.from_sha)
        to_full = prefix_to_full.get(p.to_sha, p.to_sha)
        if from_full in valid_shas and to_full in valid_shas:
            resolved.append(DiffPair(from_sha=from_full, to_sha=to_full, label=p.label))

    # Ensure first commit is a from_sha and last commit is a to_sha somewhere
    if commits and resolved:
        first_sha = commits[0].sha
        last_sha = commits[-1].sha
        has_first = any(p.from_sha == first_sha for p in resolved)
        has_last = any(p.to_sha == last_sha for p in resolved)
        if not has_first:
            # Add a pair from first commit to the earliest existing from_sha
            earliest_from = min(
                resolved,
                key=lambda p: next((i for i, c in enumerate(commits) if c.sha == p.from_sha), 0),
            )
            resolved.insert(
                0,
                DiffPair(
                    from_sha=first_sha,
                    to_sha=earliest_from.from_sha,
                    label="Project inception",
                ),
            )
        if not has_last:
            # Add a pair from the latest existing to_sha to last commit
            latest_to = max(
                resolved,
                key=lambda p: next((i for i, c in enumerate(commits) if c.sha == p.to_sha), 0),
            )
            resolved.append(
                DiffPair(
                    from_sha=latest_to.to_sha,
                    to_sha=last_sha,
                    label="Latest changes",
                ),
            )

    # Sort by chronological order of from_sha
    sha_order = {c.sha: i for i, c in enumerate(commits)}
    resolved.sort(key=lambda p: sha_order.get(p.from_sha, 0))

    # Confirm with user before proceeding to expensive analysis
    print(f"\nTriage identified {len(resolved)} diff ranges covering the repository's story:")
    for p in resolved:
        print(f"  {p.from_sha[:8]}..{p.to_sha[:8]}: {p.label}")
    answer = input("Proceed? [Y/n] ").strip().lower()
    if answer and answer != "y":
        print("Aborted by user.")
        raise SystemExit(0)

    return {"diff_pairs": resolved}
