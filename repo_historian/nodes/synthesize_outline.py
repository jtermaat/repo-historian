"""Node: synthesize_outline (deterministic, no LLM)."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from repo_historian.state import CommitAnalysis, GraphState


def synthesize_outline(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    meta = state["repo_metadata"]
    eras = state["eras"]
    analyses_by_sha: dict[str, CommitAnalysis] = {a.sha: a for a in state["commit_analyses"]}

    lines: list[str] = []
    lines.append(f"# Repository History: {meta.full_name}\n")
    lines.append(f"**{meta.description}**\n")
    lines.append(f"Stars: {meta.stars} | Forks: {meta.forks} | Language: {meta.language}\n")

    for era in eras:
        lines.append(f"\n## {era.title} ({era.start_date} — {era.end_date})\n")
        lines.append(f"_{era.description}_\n")

        for sha in era.commit_shas:
            analysis = analyses_by_sha.get(sha)
            if not analysis:
                continue
            short = sha[:8]
            commit_url = f"https://github.com/{meta.full_name}/commit/{sha}"
            tag_str = f" (tags: {', '.join(analysis.tags)})" if analysis.tags else ""
            lines.append(f"\n### [{short}]({commit_url}){tag_str} — {analysis.date}\n")
            lines.append(
                f"**+{analysis.additions} / -{analysis.deletions}** | {analysis.summary}\n"
            )
            if analysis.key_changes:
                for change in analysis.key_changes:
                    lines.append(f"- {change}")
                lines.append("")

    return {"outline": "\n".join(lines)}
