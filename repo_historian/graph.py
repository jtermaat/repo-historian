"""LangGraph graph construction."""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from repo_historian.nodes import (
    analyze_diff,
    fetch_commit_history,
    fetch_repo_metadata,
    select_analyses,
    triage_commits,
    write_narrative,
)
from repo_historian.nodes._helpers import _retry_on
from repo_historian.state import (
    CommitRecord,
    DiffAnalysisInput,
    GraphState,
)


def _fan_out_analyses(state: GraphState) -> list[Send]:
    """Generate Send() calls for each diff pair."""
    all_commits = state["all_commits"]
    meta = state["repo_metadata"]

    commit_index: dict[str, int] = {c.sha: i for i, c in enumerate(all_commits)}

    sends: list[Send] = []
    for pair in state["diff_pairs"]:
        from_sha = pair.from_sha
        to_sha = pair.to_sha

        if from_sha not in commit_index or to_sha not in commit_index:
            continue

        from_idx = commit_index[from_sha]
        to_idx = commit_index[to_sha]

        from_commit = all_commits[from_idx]
        to_commit = all_commits[to_idx]
        commits_in_range: list[CommitRecord] = all_commits[from_idx : to_idx + 1]

        inp: DiffAnalysisInput = {
            "from_sha": from_sha,
            "to_sha": to_sha,
            "repo_full_name": meta.full_name,
            "from_commit": from_commit,
            "to_commit": to_commit,
            "commits_in_range": commits_in_range,
            "label": pair.label,
        }
        sends.append(Send("analyze_diff", inp))

    return sends


def build_graph() -> StateGraph:
    """Build and compile the Repo Historian graph."""
    graph = StateGraph(GraphState)
    retry = RetryPolicy(retry_on=_retry_on)

    graph.add_node("fetch_repo_metadata", fetch_repo_metadata, retry=retry)
    graph.add_node("fetch_commit_history", fetch_commit_history, retry=retry)
    graph.add_node("triage_commits", triage_commits, retry=retry)
    graph.add_node("analyze_diff", analyze_diff, retry=retry)
    graph.add_node("select_analyses", select_analyses, retry=retry)
    graph.add_node("write_narrative", write_narrative, retry=retry)

    graph.add_edge(START, "fetch_repo_metadata")
    graph.add_edge("fetch_repo_metadata", "fetch_commit_history")
    graph.add_edge("fetch_commit_history", "triage_commits")
    graph.add_conditional_edges("triage_commits", _fan_out_analyses, ["analyze_diff"])
    graph.add_edge("analyze_diff", "select_analyses")
    graph.add_edge("select_analyses", "write_narrative")
    graph.add_edge("write_narrative", END)

    return graph.compile()


def build_per_repo_graph():
    """Build a per-repo subgraph that stops after analyze_diff.

    Used by the multi-repo orchestrator to run the per-repo pipeline
    without narrative generation (that happens at the orchestrator level).
    """
    graph = StateGraph(GraphState)
    retry = RetryPolicy(retry_on=_retry_on)

    graph.add_node("fetch_repo_metadata", fetch_repo_metadata, retry=retry)
    graph.add_node("fetch_commit_history", fetch_commit_history, retry=retry)
    graph.add_node("triage_commits", triage_commits, retry=retry)
    graph.add_node("analyze_diff", analyze_diff, retry=retry)
    graph.add_node("select_analyses", select_analyses, retry=retry)

    graph.add_edge(START, "fetch_repo_metadata")
    graph.add_edge("fetch_repo_metadata", "fetch_commit_history")
    graph.add_edge("fetch_commit_history", "triage_commits")
    graph.add_conditional_edges("triage_commits", _fan_out_analyses, ["analyze_diff"])
    graph.add_edge("analyze_diff", "select_analyses")
    graph.add_edge("select_analyses", END)

    return graph.compile()
