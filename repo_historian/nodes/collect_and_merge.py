"""Node: collect_and_merge — merge per-repo results for cross-repo synthesis."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from repo_historian.state import MultiRepoGraphState, RepoAnalysisResult


def collect_and_merge(state: MultiRepoGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Merge per-repo results and prompt for confirmation."""
    repo_results: list[RepoAnalysisResult] = state["repo_results"]

    all_metadata = [r.repo_metadata for r in repo_results]
    all_analyses: list = []

    print(f"\n{'=' * 60}")
    print("Cross-repo triage summary:")
    print(f"{'=' * 60}")

    for r in repo_results:
        print(f"\n  {r.repo_metadata.full_name}:")
        print(f"    {len(r.diff_analyses)} diff ranges analyzed")
        for dp in r.diff_pairs:
            print(f"      {dp.from_sha[:8]}..{dp.to_sha[:8]}: {dp.label}")

        all_analyses.extend(r.diff_analyses)

    # Sort chronologically
    all_analyses.sort(key=lambda a: a.start_date)

    print(f"\n  Total: {len(all_analyses)} diff ranges across {len(repo_results)} repos")
    print(f"{'=' * 60}")

    skip_confirmation = config.get("configurable", {}).get("skip_confirmation", False)
    if not skip_confirmation:
        answer = input("Proceed with cross-repo synthesis? [Y/n] ").strip().lower()
        if answer and answer != "y":
            print("Aborted by user.")
            raise SystemExit(0)

    return {
        "all_repo_metadata": all_metadata,
        "merged_analyses": all_analyses,
    }
