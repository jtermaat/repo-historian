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


@dataclass
class Era:
    title: str
    start_date: str
    end_date: str
    description: str
    diff_pair_keys: list[str]


@dataclass
class TriageConfig:
    batch_size: int


# --- Substate for analyze_diff fan-out via Send() ---


class DiffAnalysisInput(TypedDict):
    from_sha: str
    to_sha: str
    repo_full_name: str
    from_commit: CommitRecord
    to_commit: CommitRecord
    commits_in_range: list[CommitRecord]
    label: str


# --- LangGraph top-level state ---


class GraphState(TypedDict):
    repo_url: str
    triage_config: TriageConfig
    repo_metadata: RepoMetadata
    all_commits: list[CommitRecord]
    diff_pairs: list[DiffPair]
    diff_analyses: Annotated[list[DiffAnalysis], operator.add]
    eras: list[Era]
    outline: str
    narrative: str
