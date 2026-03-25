"""Multi-repo orchestrator graph."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from repo_historian.nodes import (
    collect_and_merge,
    cross_repo_cluster_eras,
    cross_repo_expand_narrative,
    cross_repo_synthesize_outline,
    run_single_repo,
)
from repo_historian.nodes._helpers import _retry_on
from repo_historian.state import MultiRepoGraphState, PerRepoInput

# --- Conditional edge: fan out to per-repo pipelines ---


def _fan_out_repos(state: MultiRepoGraphState, config: RunnableConfig) -> list[Send]:
    """Generate Send() calls for each repo URL."""
    github_token = config.get("configurable", {}).get("github_token", "")
    sends: list[Send] = []
    for url in state["repo_urls"]:
        inp: PerRepoInput = {
            "repo_url": url,
            "github_token": github_token,
        }
        sends.append(Send("run_single_repo", inp))
    return sends


# --- Graph builder ---


def build_multi_repo_graph():
    """Build and compile the multi-repo orchestrator graph."""
    graph = StateGraph(MultiRepoGraphState)
    retry = RetryPolicy(retry_on=_retry_on)

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
