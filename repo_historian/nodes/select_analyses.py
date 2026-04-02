"""Node: select_analyses — pick the most representative diff analyses."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repo_historian import logger
from repo_historian.config import MAX_SELECTED_ANALYSES
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import GraphState


class _SelectionResponse(BaseModel):
    selected_ids: list[int]


def select_analyses(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Select the most representative analyses for narrative generation."""
    analyses = state["diff_analyses"]
    meta = state["repo_metadata"]

    if len(analyses) <= MAX_SELECTED_ANALYSES:
        logger.info(
            "Keeping all %d analyses for %s (within cap of %d)",
            len(analyses),
            meta.full_name,
            MAX_SELECTED_ANALYSES,
        )
        return {"selected_analyses": list(analyses)}

    # Build minimal context: ID + label + key_changes only
    lines: list[str] = []
    for i, a in enumerate(analyses):
        changes = "\n".join(f"- {c}" for c in a.key_changes)
        lines.append(f"ID {i} | {a.label}\n{changes}")

    llm = build_llm()
    structured_llm = llm.with_structured_output(_SelectionResponse)

    result: _SelectionResponse = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software-history curator. Given a numbered list of diff "
                    f"analyses from {meta.full_name}, select up to "
                    f"{MAX_SELECTED_ANALYSES} that best represent the project's "
                    "evolution and would be most relevant to a narrative piece about "
                    "this repository.\n\n"
                    "Prefer analyses that reveal:\n"
                    "- Major architectural shifts\n"
                    "- New capabilities or feature areas\n"
                    "- Significant milestones or releases\n"
                    "- Turning points in project direction\n\n"
                    "The selected set should be representative of the full arc of the "
                    "project's history. Return only the numeric IDs."
                )
            ),
            HumanMessage(content="\n\n".join(lines)),
        ]
    )

    # Filter: deduplicate, skip out-of-range, preserve original order
    valid_range = range(len(analyses))
    seen: set[int] = set()
    kept_ids: list[int] = []
    for idx in result.selected_ids:
        if idx in valid_range and idx not in seen:
            seen.add(idx)
            kept_ids.append(idx)

    kept_ids.sort()
    selected = [analyses[i] for i in kept_ids]

    logger.info(
        "Selected %d of %d analyses for %s",
        len(selected),
        len(analyses),
        meta.full_name,
    )

    return {"selected_analyses": selected}
