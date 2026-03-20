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
    commit_shas: list[str]


class _ErasResponse(BaseModel):
    eras: list[_EraItem] = Field(min_length=MIN_ERAS, max_length=MAX_ERAS)


def cluster_into_eras(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    analyses = state["commit_analyses"]
    # Sort chronologically by date
    analyses_sorted = sorted(analyses, key=lambda a: a.date)

    summaries = "\n".join(
        f"- {a.sha[:8]} ({a.date}): era_hint={a.era_hint!r}, summary={a.summary}"
        for a in analyses_sorted
    )

    llm = build_llm()
    structured_llm = llm.with_structured_output(_ErasResponse)

    result: _ErasResponse = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are a software historian. Group the following commit analyses into "
                    f"{MIN_ERAS}-{MAX_ERAS} thematic eras that tell the story of the project's "
                    f"evolution. Each era needs a title, start/end dates, description, and the "
                    f"list of commit SHAs (full 8-char prefixes) belonging to it. "
                    f"Every commit must belong to exactly one era."
                )
            ),
            HumanMessage(content=f"Commit analyses:\n{summaries}"),
        ]
    )

    # Map 8-char prefixes back to full SHAs
    sha_lookup: dict[str, str] = {a.sha[:8]: a.sha for a in analyses}

    eras: list[Era] = []
    for e in result.eras:
        full_shas = [sha_lookup.get(s[:8], s) for s in e.commit_shas]
        eras.append(
            Era(
                title=e.title,
                start_date=e.start_date,
                end_date=e.end_date,
                description=e.description,
                commit_shas=full_shas,
            )
        )

    return {"eras": eras}
