"""Node: synthesize_outline (deterministic, no LLM)."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from repo_historian import logger
from repo_historian.state import DiffAnalysis, GraphState


def synthesize_outline(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    meta = state["repo_metadata"]
    eras = state["eras"]
    logger.info("Building outline across %d eras", len(eras))
    analyses_by_key: dict[str, DiffAnalysis] = {a.pair_key: a for a in state["diff_analyses"]}

    lines: list[str] = []
    lines.append(f"# Repository History: {meta.full_name}\n")
    lines.append(f"**{meta.description}**\n")
    lines.append(f"Stars: {meta.stars} | Forks: {meta.forks} | Language: {meta.language}\n")

    for era in eras:
        lines.append(f"\n## {era.title} ({era.start_date} — {era.end_date})\n")
        lines.append(f"_{era.description}_\n")

        for pair_key in era.diff_pair_keys:
            analysis = analyses_by_key.get(pair_key)
            if not analysis:
                continue
            short = f"{analysis.from_sha[:8]}..{analysis.to_sha[:8]}"
            compare_url = (
                f"https://github.com/{meta.full_name}/compare/"
                f"{analysis.from_sha}...{analysis.to_sha}"
            )
            tag_str = f" (tags: {', '.join(analysis.tags)})" if analysis.tags else ""
            lines.append(f"\n### [{short}]({compare_url}){tag_str}")
            lines.append(
                f"**{analysis.start_date} — {analysis.end_date}** | "
                f"{analysis.commit_count} commits | "
                f"+{analysis.additions} / -{analysis.deletions}\n"
            )
            lines.append(f"{analysis.summary}\n")
            if analysis.key_changes:
                for change in analysis.key_changes:
                    lines.append(f"- {change}")
                lines.append("")

    return {"outline": "\n".join(lines)}
