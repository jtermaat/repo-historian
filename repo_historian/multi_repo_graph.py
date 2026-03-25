"""Multi-repo orchestrator graph."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, default_retry_on

from repo_historian import logger
from repo_historian.graph import build_per_repo_graph
from repo_historian.nodes.cross_repo_cluster_eras import cross_repo_cluster_eras
from repo_historian.nodes.cross_repo_expand_narrative import (
    cross_repo_expand_narrative,
)
from repo_historian.nodes.cross_repo_synthesize_outline import (
    cross_repo_synthesize_outline,
)
from repo_historian.state import MultiRepoGraphState, RepoAnalysisResult

# --- Substate for per-repo fan-out ---


class _PerRepoInput(TypedDict):
    repo_url: str
    github_token: str


def _retry_on(exc: Exception) -> bool:
    from openai import LengthFinishReasonError

    if isinstance(exc, LengthFinishReasonError):
        return False
    return default_retry_on(exc)


# --- Node: run a single repo through the per-repo pipeline ---


def run_single_repo(state: _PerRepoInput, config: RunnableConfig) -> dict[str, Any]:
    """Run the per-repo pipeline for one repository."""
    repo_url = state["repo_url"]
    github_token = state["github_token"]

    logger.info("Starting per-repo pipeline for %s", repo_url)

    per_repo_graph = build_per_repo_graph()
    result = per_repo_graph.invoke(
        {"repo_url": repo_url},
        config={
            "configurable": {
                "github_token": github_token,
                "skip_confirmation": True,
            },
        },
    )

    repo_result = RepoAnalysisResult(
        repo_url=repo_url,
        repo_metadata=result["repo_metadata"],
        all_commits=result["all_commits"],
        diff_pairs=result["diff_pairs"],
        diff_analyses=result["diff_analyses"],
        batch_summaries=result.get("batch_summaries", []),
    )

    logger.info(
        "Completed per-repo pipeline for %s: %d analyses, %d batch summaries",
        repo_url,
        len(repo_result.diff_analyses),
        len(repo_result.batch_summaries),
    )

    return {"repo_results": [repo_result]}


# --- Conditional edge: fan out to per-repo pipelines ---


def _fan_out_repos(state: MultiRepoGraphState, config: RunnableConfig) -> list[Send]:
    """Generate Send() calls for each repo URL."""
    github_token = config.get("configurable", {}).get("github_token", "")
    sends: list[Send] = []
    for url in state["repo_urls"]:
        inp: _PerRepoInput = {
            "repo_url": url,
            "github_token": github_token,
        }
        sends.append(Send("run_single_repo", inp))
    return sends


# --- Node: collect and merge per-repo results ---


def collect_and_merge(state: MultiRepoGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Merge per-repo results and prompt for confirmation."""
    repo_results: list[RepoAnalysisResult] = state["repo_results"]

    all_metadata = [r.repo_metadata for r in repo_results]
    all_analyses: list = []
    all_batch_summaries: list = []

    print(f"\n{'=' * 60}")
    print("Cross-repo triage summary:")
    print(f"{'=' * 60}")

    for r in repo_results:
        print(f"\n  {r.repo_metadata.full_name}:")
        print(f"    {len(r.diff_analyses)} diff ranges analyzed")
        print(f"    {len(r.batch_summaries)} batch summaries")
        for dp in r.diff_pairs:
            print(f"      {dp.from_sha[:8]}..{dp.to_sha[:8]}: {dp.label}")

        all_analyses.extend(r.diff_analyses)
        all_batch_summaries.extend(r.batch_summaries)

    # Sort chronologically
    all_analyses.sort(key=lambda a: a.start_date)
    all_batch_summaries.sort(key=lambda bs: bs.start_date)

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
        "merged_batch_summaries": all_batch_summaries,
    }


# --- Graph builder ---


def build_multi_repo_graph():
    """Build and compile the multi-repo orchestrator graph."""
    graph = StateGraph(MultiRepoGraphState)
    retry = RetryPolicy(retry_on=_retry_on)

    # fan_out_repos is a normal node whose output triggers Send() to run_single_repo
    graph.add_node("run_single_repo", run_single_repo)
    graph.add_node("collect_and_merge", collect_and_merge)
    graph.add_node("cross_repo_cluster_eras", cross_repo_cluster_eras, retry=retry)
    graph.add_node("cross_repo_synthesize_outline", cross_repo_synthesize_outline)
    graph.add_node("cross_repo_expand_narrative", cross_repo_expand_narrative, retry=retry)

    # START → fan out via conditional edge → parallel run_single_repo nodes
    graph.add_conditional_edges(START, _fan_out_repos, ["run_single_repo"])
    graph.add_edge("run_single_repo", "collect_and_merge")
    graph.add_edge("collect_and_merge", "cross_repo_cluster_eras")
    graph.add_edge("cross_repo_cluster_eras", "cross_repo_synthesize_outline")
    graph.add_edge("cross_repo_synthesize_outline", "cross_repo_expand_narrative")
    graph.add_edge("cross_repo_expand_narrative", END)

    return graph.compile()
