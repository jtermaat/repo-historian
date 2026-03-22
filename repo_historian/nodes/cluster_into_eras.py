"""Node: cluster_into_eras."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from repo_historian.config import MAX_ERAS, MIN_ERAS
from repo_historian.nodes._helpers import build_llm
from repo_historian.state import Era, GraphState


class _EraItem(BaseModel):
    title: str
    start_date: str
    end_date: str
    description: str
    diff_pair_keys: list[str]


class _ErasResponse(BaseModel):
    eras: list[_EraItem] = Field(min_length=MIN_ERAS, max_length=MAX_ERAS)


def cluster_into_eras(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    analyses = state["diff_analyses"]
    # Sort chronologically by start date
    analyses_sorted = sorted(analyses, key=lambda a: a.start_date)

    summaries = "\n".join(
        f"- {a.from_sha[:8]}..{a.to_sha[:8]} ({a.start_date} to {a.end_date}): "
        f"era_hint={a.era_hint!r}, summary={a.summary}"
        for a in analyses_sorted
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_ErasResponse)

    result: _ErasResponse = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are a software historian. Group the following diff-pair analyses into "
                    f"{MIN_ERAS}-{MAX_ERAS} thematic eras that tell the story of the project's "
                    f"evolution. Each era needs a title, start/end dates, description, and the "
                    f"list of diff pair keys (in from..to format using 8-char SHA prefixes as "
                    f"shown) belonging to it. Every diff pair must belong to exactly one era."
                )
            ),
            HumanMessage(content=f"Diff-pair analyses:\n{summaries}"),
        ]
    )

    # Map short pair keys back to full pair keys
    short_to_full: dict[str, str] = {
        f"{a.from_sha[:8]}..{a.to_sha[:8]}": a.pair_key for a in analyses
    }

    eras: list[Era] = []
    for e in result.eras:
        full_keys = [short_to_full.get(k, k) for k in e.diff_pair_keys]
        eras.append(
            Era(
                title=e.title,
                start_date=e.start_date,
                end_date=e.end_date,
                description=e.description,
                diff_pair_keys=full_keys,
            )
        )

    return {"eras": eras}
