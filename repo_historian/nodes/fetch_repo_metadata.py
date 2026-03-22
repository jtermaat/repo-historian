"""Node: fetch_repo_metadata."""

from __future__ import annotations

from typing import Any

from github import Github
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from repo_historian import logger
from repo_historian.nodes._helpers import get_github_token, parse_repo_full_name
from repo_historian.state import GraphState, RepoMetadata


@traceable(run_type="tool", name="fetch_repo_metadata")
def fetch_repo_metadata(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    full_name = parse_repo_full_name(state["repo_url"])
    logger.info("Fetching metadata for %s", full_name)
    g = Github(get_github_token(config))
    repo = g.get_repo(full_name)

    metadata = RepoMetadata(
        full_name=repo.full_name,
        description=repo.description or "",
        stars=repo.stargazers_count,
        forks=repo.forks_count,
        language=repo.language or "Unknown",
        topics=repo.get_topics(),
        html_url=repo.html_url,
    )
    return {"repo_metadata": metadata}
