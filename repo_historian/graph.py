"""LangGraph graph construction."""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from repo_historian.config import CONTEXT_WINDOW_COMMITS
from repo_historian.nodes import (
    analyze_commit,
    cluster_into_eras,
    expand_to_narrative,
    fetch_commit_history,
    fetch_repo_metadata,
    synthesize_outline,
    triage_commits,
)
from repo_historian.state import (
    CommitAnalysisInput,
    CommitRecord,
    GraphState,
)


def _fan_out_analyses(state: GraphState) -> list[Send]:
    """Generate Send() calls for each significant commit."""
    all_commits = state["all_commits"]
    meta = state["repo_metadata"]

    commit_index: dict[str, int] = {c.sha: i for i, c in enumerate(all_commits)}

    sends: list[Send] = []
    for sha in state["significant_commit_shas"]:
        if sha not in commit_index:
            continue
        idx = commit_index[sha]
        commit_record = all_commits[idx]

        # Gather context window
        start = max(0, idx - CONTEXT_WINDOW_COMMITS)
        end = min(len(all_commits), idx + CONTEXT_WINDOW_COMMITS + 1)
        context: list[CommitRecord] = [all_commits[j] for j in range(start, end) if j != idx]

        inp: CommitAnalysisInput = {
            "sha": sha,
            "repo_full_name": meta.full_name,
            "commit_record": commit_record,
            "context_commits": context,
        }
        sends.append(Send("analyze_commit", inp))

    return sends


def build_graph() -> StateGraph:
    """Build and compile the Repo Historian graph."""
    graph = StateGraph(GraphState)

    graph.add_node("fetch_repo_metadata", fetch_repo_metadata, retry=RetryPolicy())
    graph.add_node("fetch_commit_history", fetch_commit_history, retry=RetryPolicy())
    graph.add_node("triage_commits", triage_commits, retry=RetryPolicy())
    graph.add_node("analyze_commit", analyze_commit, retry=RetryPolicy())
    graph.add_node("cluster_into_eras", cluster_into_eras, retry=RetryPolicy())
    graph.add_node("synthesize_outline", synthesize_outline)
    graph.add_node("expand_to_narrative", expand_to_narrative, retry=RetryPolicy())

    graph.add_edge(START, "fetch_repo_metadata")
    graph.add_edge("fetch_repo_metadata", "fetch_commit_history")
    graph.add_edge("fetch_commit_history", "triage_commits")
    graph.add_conditional_edges("triage_commits", _fan_out_analyses, ["analyze_commit"])
    graph.add_edge("analyze_commit", "cluster_into_eras")
    graph.add_edge("cluster_into_eras", "synthesize_outline")
    graph.add_edge("synthesize_outline", "expand_to_narrative")
    graph.add_edge("expand_to_narrative", END)

    return graph.compile()
