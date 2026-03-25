"""LangGraph graph construction."""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, default_retry_on

from repo_historian.nodes import (
    analyze_diff,
    cluster_into_eras,
    expand_to_narrative,
    fetch_commit_history,
    fetch_repo_metadata,
    summarize_analyses,
    synthesize_outline,
    triage_commits,
)
from repo_historian.state import (
    CommitRecord,
    DiffAnalysisInput,
    GraphState,
)


def _retry_on(exc: Exception) -> bool:
    """Don't retry length-limit errors — the same prompt will fail again."""
    from openai import LengthFinishReasonError

    if isinstance(exc, LengthFinishReasonError):
        return False
    return default_retry_on(exc)


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
    graph.add_node("summarize_analyses", summarize_analyses, retry=retry)
    graph.add_node("cluster_into_eras", cluster_into_eras, retry=retry)
    graph.add_node("synthesize_outline", synthesize_outline)
    graph.add_node("expand_to_narrative", expand_to_narrative, retry=retry)

    graph.add_edge(START, "fetch_repo_metadata")
    graph.add_edge("fetch_repo_metadata", "fetch_commit_history")
    graph.add_edge("fetch_commit_history", "triage_commits")
    graph.add_conditional_edges("triage_commits", _fan_out_analyses, ["analyze_diff"])
    graph.add_edge("analyze_diff", "summarize_analyses")
    graph.add_edge("summarize_analyses", "cluster_into_eras")
    graph.add_edge("cluster_into_eras", "synthesize_outline")
    graph.add_edge("synthesize_outline", "expand_to_narrative")
    graph.add_edge("expand_to_narrative", END)

    return graph.compile()
