"""Graph state and dataclass definitions."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict


@dataclass
class RepoMetadata:
    full_name: str
    description: str
    stars: int
    forks: int
    language: str
    topics: list[str]
    html_url: str


@dataclass
class CommitRecord:
    sha: str
    message: str
    author: str
    date: str  # ISO format
    tags: list[str] = field(default_factory=list)


@dataclass
class InflectionPoint:
    sha: str
    label: str


@dataclass
class DiffPair:
    from_sha: str
    to_sha: str
    label: str


@dataclass
class DiffAnalysis:
    pair_key: str  # "{from_sha}..{to_sha}"
    from_sha: str
    to_sha: str
    era_hint: str
    summary: str
    narrative_paragraph: str
    key_changes: list[str]
    additions: int = 0
    deletions: int = 0
    start_date: str = ""
    end_date: str = ""
    label: str = ""
    commit_count: int = 0
    from_message: str = ""
    to_message: str = ""
    authors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    repo_full_name: str = ""


# --- Substate for analyze_diff fan-out via Send() ---


class DiffAnalysisInput(TypedDict):
    from_sha: str
    to_sha: str
    repo_full_name: str
    from_commit: CommitRecord
    to_commit: CommitRecord
    commits_in_range: list[CommitRecord]
    label: str


class PerRepoInput(TypedDict):
    repo_url: str
    github_token: str


# --- LangGraph top-level state ---


class GraphState(TypedDict):
    repo_url: str
    repo_metadata: RepoMetadata
    all_commits: list[CommitRecord]
    diff_pairs: list[DiffPair]
    diff_analyses: Annotated[list[DiffAnalysis], operator.add]
    narrative: str


# --- Multi-repo orchestrator state ---


@dataclass
class RepoAnalysisResult:
    """Collected output from one per-repo pipeline run."""

    repo_url: str
    repo_metadata: RepoMetadata
    all_commits: list[CommitRecord]
    diff_pairs: list[DiffPair]
    diff_analyses: list[DiffAnalysis]


class MultiRepoGraphState(TypedDict):
    repo_urls: list[str]
    repo_results: Annotated[list[RepoAnalysisResult], operator.add]
    all_repo_metadata: list[RepoMetadata]
    merged_analyses: list[DiffAnalysis]
    narrative: str
