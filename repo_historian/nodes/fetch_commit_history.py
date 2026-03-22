"""Node: fetch_commit_history."""

from __future__ import annotations

from typing import Any

from github import Github
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from repo_historian import logger
from repo_historian.nodes._helpers import get_github_token
from repo_historian.state import CommitRecord, GraphState


@traceable(run_type="tool", name="fetch_commit_history")
def fetch_commit_history(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    g = Github(get_github_token(config))
    full_name = state["repo_metadata"].full_name
    repo = g.get_repo(full_name)

    # Build tag map: sha -> [tag_name, ...]
    tag_map: dict[str, list[str]] = {}
    for tag in repo.get_tags():
        tag_map.setdefault(tag.commit.sha, []).append(tag.name)

    commits_page = repo.get_commits()
    records: list[CommitRecord] = []
    for c in commits_page:
        records.append(
            CommitRecord(
                sha=c.sha,
                message=c.commit.message.split("\n")[0],  # first line only
                author=c.commit.author.name if c.commit.author else "Unknown",
                date=c.commit.author.date.isoformat() if c.commit.author else "",
                tags=tag_map.get(c.sha, []),
            )
        )

    # Chronological (oldest first)
    records.reverse()
    logger.info("Found %d commits, %d tagged", len(records), len(tag_map))
    return {"all_commits": records}
