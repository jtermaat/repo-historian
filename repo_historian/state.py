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
class TriageScore:
    sha: str
    focus: bool
    reason: str


@dataclass
class CommitAnalysis:
    sha: str
    era_hint: str
    summary: str
    narrative_paragraph: str
    key_changes: list[str]
    additions: int = 0
    deletions: int = 0
    date: str = ""
    message: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Era:
    title: str
    start_date: str
    end_date: str
    description: str
    commit_shas: list[str]


@dataclass
class TriageConfig:
    batch_size: int


# --- Substate for analyze_commit fan-out via Send() ---


class CommitAnalysisInput(TypedDict):
    sha: str
    repo_full_name: str
    commit_record: CommitRecord
    context_commits: list[CommitRecord]


# --- LangGraph top-level state ---


class GraphState(TypedDict):
    repo_url: str
    triage_config: TriageConfig
    repo_metadata: RepoMetadata
    all_commits: list[CommitRecord]
    significant_commit_shas: list[str]
    commit_analyses: Annotated[list[CommitAnalysis], operator.add]
    eras: list[Era]
    outline: str
    narrative: str
