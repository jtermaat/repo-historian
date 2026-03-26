"""Node: run_single_repo — run the per-repo pipeline for one repository."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from repo_historian import logger
from repo_historian.state import PerRepoInput, RepoAnalysisResult


def run_single_repo(state: PerRepoInput, config: RunnableConfig) -> dict[str, Any]:
    """Run the per-repo pipeline for one repository."""
    from repo_historian.graph import build_per_repo_graph

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
    )

    logger.info(
        "Completed per-repo pipeline for %s: %d analyses",
        repo_url,
        len(repo_result.diff_analyses),
    )

    return {"repo_results": [repo_result]}
