"""Node: cross_repo_synthesize_outline (deterministic, no LLM)."""

from __future__ import annotations

from itertools import groupby
from typing import Any

from langchain_core.runnables import RunnableConfig

from repo_historian import logger
from repo_historian.state import DiffAnalysis, MultiRepoGraphState, RepoMetadata


def cross_repo_synthesize_outline(
    state: MultiRepoGraphState, config: RunnableConfig
) -> dict[str, Any]:
    all_metadata: list[RepoMetadata] = state["all_repo_metadata"]
    eras = state["cross_repo_eras"]
    analyses = state["merged_analyses"]
    logger.info("Building cross-repo outline across %d eras", len(eras))

    analyses_by_key: dict[str, DiffAnalysis] = {a.pair_key: a for a in analyses}

    # Derive ecosystem name from shared org or list of repos
    repo_names = [m.full_name for m in all_metadata]
    orgs = {name.split("/")[0] for name in repo_names}
    if len(orgs) == 1:
        ecosystem_name = f"{next(iter(orgs))} Ecosystem"
    else:
        ecosystem_name = " + ".join(repo_names)

    lines: list[str] = []
    lines.append(f"# Ecosystem History: {ecosystem_name}\n")

    # List all repos
    for meta in all_metadata:
        lines.append(
            f"- **[{meta.full_name}]({meta.html_url})**: {meta.description}  "
            f"Stars: {meta.stars} | Language: {meta.language}"
        )
    lines.append("")

    for era in eras:
        lines.append(f"\n## {era.title} ({era.start_date} — {era.end_date})\n")
        lines.append(f"_{era.description}_\n")

        # Collect analyses for this era, grouped by repo
        era_analyses: list[DiffAnalysis] = []
        for pair_key in era.diff_pair_keys:
            a = analyses_by_key.get(pair_key)
            if a:
                era_analyses.append(a)

        era_analyses.sort(key=lambda a: a.repo_full_name)
        for repo_name, group in groupby(era_analyses, key=lambda a: a.repo_full_name):
            lines.append(f"\n#### {repo_name}\n")
            for analysis in group:
                short = f"{analysis.from_sha[:8]}..{analysis.to_sha[:8]}"
                compare_url = (
                    f"https://github.com/{analysis.repo_full_name}/compare/"
                    f"{analysis.from_sha}...{analysis.to_sha}"
                )
                tag_str = f" (tags: {', '.join(analysis.tags)})" if analysis.tags else ""
                lines.append(f"\n##### [{short}]({compare_url}){tag_str}")
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
