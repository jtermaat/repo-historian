"""Graph node implementations."""

from __future__ import annotations

import re
from typing import Any

from github import Github
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from repo_historian.config import (
    MAX_DIFF_CHARS_TOTAL,
    MAX_ERAS,
    MAX_FILES_PER_COMMIT,
    MAX_NARRATIVE_WORDS,
    MAX_PATCH_CHARS_PER_FILE,
    MIN_ERAS,
    MIN_NARRATIVE_WORDS,
    MODEL_NAME,
    detect_provider,
)
from repo_historian.state import (
    CommitAnalysis,
    CommitAnalysisInput,
    CommitRecord,
    Era,
    GraphState,
    RepoMetadata,
    TriageScore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_repo_full_name(url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    match = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ValueError(f"Invalid GitHub repo URL: {url}")
    return match.group(1)


def _get_github_token(config: RunnableConfig) -> str:
    return config["configurable"]["github_token"]


def _build_llm() -> BaseChatModel:
    provider = detect_provider(MODEL_NAME)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=MODEL_NAME)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=MODEL_NAME)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=MODEL_NAME)
    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class _TriageScoreItem(BaseModel):
    sha: str
    score: int
    reason: str


class _TriageResponse(BaseModel):
    scores: list[_TriageScoreItem]


class _AnalysisOutput(BaseModel):
    era_hint: str
    summary: str
    narrative_paragraph: str
    key_changes: list[str]


class _EraItem(BaseModel):
    title: str
    start_date: str
    end_date: str
    description: str
    commit_shas: list[str]


class _ErasResponse(BaseModel):
    eras: list[_EraItem] = Field(min_length=MIN_ERAS, max_length=MAX_ERAS)


# ---------------------------------------------------------------------------
# Node 1: fetch_repo_metadata
# ---------------------------------------------------------------------------


def fetch_repo_metadata(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    g = Github(_get_github_token(config))
    full_name = _parse_repo_full_name(state["repo_url"])
    repo = g.get_repo(full_name)

    metadata = RepoMetadata(
        full_name=repo.full_name,
        description=repo.description or "",
        stars=repo.stargazers_count,
        forks=repo.forks_count,
        language=repo.language or "Unknown",
        topics=repo.get_topics(),
        html_url=repo.html_url,
    )
    return {"repo_metadata": metadata}


# ---------------------------------------------------------------------------
# Node 2: fetch_commit_history
# ---------------------------------------------------------------------------


def fetch_commit_history(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    g = Github(_get_github_token(config))
    full_name = state["repo_metadata"].full_name
    repo = g.get_repo(full_name)

    # Build tag map: sha → [tag_name, ...]
    tag_map: dict[str, list[str]] = {}
    for tag in repo.get_tags():
        tag_map.setdefault(tag.commit.sha, []).append(tag.name)

    commits_page = repo.get_commits()
    records: list[CommitRecord] = []
    for c in commits_page:
        records.append(
            CommitRecord(
                sha=c.sha,
                message=c.commit.message.split("\n")[0],  # first line only
                author=c.commit.author.name if c.commit.author else "Unknown",
                date=c.commit.author.date.isoformat() if c.commit.author else "",
                tags=tag_map.get(c.sha, []),
            )
        )

    # Chronological (oldest first)
    records.reverse()
    return {"all_commits": records}


# ---------------------------------------------------------------------------
# Node 3: triage_commits
# ---------------------------------------------------------------------------


def triage_commits(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    commits = state["all_commits"]
    triage_cfg = state["triage_config"]
    top_k = triage_cfg.top_k
    batch_size = triage_cfg.batch_size

    # Always-include SHAs: first, last, all tagged
    always_include: set[str] = set()
    if commits:
        always_include.add(commits[0].sha)
        always_include.add(commits[-1].sha)
    for c in commits:
        if c.tags:
            always_include.add(c.sha)

    # Score in batches
    llm = _build_llm()
    structured_llm = llm.with_structured_output(_TriageResponse)
    all_scores: list[TriageScore] = []

    for i in range(0, len(commits), batch_size):
        batch = commits[i : i + batch_size]
        commit_lines = "\n".join(f"- {c.sha[:8]} ({c.date}): {c.message}" for c in batch)

        result: _TriageResponse = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a commit triage assistant. Score each commit 1-10 based on "
                        "how historically significant its COMMIT MESSAGE suggests it is. "
                        "10 = major feature / breaking change / architecture shift. "
                        "1 = trivial typo / whitespace."
                    )
                ),
                HumanMessage(content=f"Score these commits:\n{commit_lines}"),
            ]
        )

        for item in result.scores:
            all_scores.append(TriageScore(sha=item.sha, score=item.score, reason=item.reason))

    # Build sha-prefix → full-sha lookup
    prefix_to_full: dict[str, str] = {}
    for c in commits:
        prefix_to_full[c.sha[:8]] = c.sha

    # Rank by score, take top-K, then merge always_include
    scored_map: dict[str, int] = {}
    for s in all_scores:
        full_sha = prefix_to_full.get(s.sha, s.sha)
        scored_map[full_sha] = s.score

    ranked = sorted(scored_map.keys(), key=lambda sha: scored_map.get(sha, 0), reverse=True)
    selected: set[str] = set(ranked[:top_k])
    selected |= always_include

    # Filter to only SHAs that actually exist in our commit list
    valid_shas = {c.sha for c in commits}
    selected &= valid_shas

    return {
        "significant_commit_shas": sorted(
            selected, key=lambda s: next((i for i, c in enumerate(commits) if c.sha == s), 0)
        )
    }


# ---------------------------------------------------------------------------
# Node 4: analyze_commit (fan-out target via Send)
# ---------------------------------------------------------------------------


def analyze_commit(state: CommitAnalysisInput, config: RunnableConfig) -> dict[str, Any]:
    """Analyze a single commit — called via Send() fan-out."""
    g = Github(_get_github_token(config))
    repo = g.get_repo(state["repo_full_name"])
    commit = repo.get_commit(state["sha"])

    # Fetch diff with caps
    diff_parts: list[str] = []
    total_chars = 0
    additions = 0
    deletions = 0

    for idx, f in enumerate(commit.files or []):
        if idx >= MAX_FILES_PER_COMMIT:
            break
        additions += f.additions
        deletions += f.deletions
        patch = f.patch or ""
        if len(patch) > MAX_PATCH_CHARS_PER_FILE:
            patch = patch[:MAX_PATCH_CHARS_PER_FILE] + "\n... [truncated]"
        if total_chars + len(patch) > MAX_DIFF_CHARS_TOTAL:
            diff_parts.append(f"--- {f.filename} [diff truncated due to total size cap]")
            break
        diff_parts.append(f"--- {f.filename}\n{patch}")
        total_chars += len(patch)

    diff_text = "\n\n".join(diff_parts) if diff_parts else "(no diff available)"

    # Context commits summary
    context_lines = "\n".join(
        f"- {c.sha[:8]} ({c.date}): {c.message}" for c in state["context_commits"]
    )

    llm = _build_llm()
    structured_llm = llm.with_structured_output(_AnalysisOutput)

    result: _AnalysisOutput = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software historian analyzing a single commit. "
                    "Provide: era_hint (a short thematic label for this period of the project), "
                    "summary (1-2 sentences), narrative_paragraph (3-5 sentences telling the "
                    "story of this change in past tense), and key_changes (3-5 bullet points)."
                )
            ),
            HumanMessage(
                content=(
                    f"Commit: {state['sha'][:8]}\n"
                    f"Message: {state['commit_record'].message}\n"
                    f"Author: {state['commit_record'].author}\n"
                    f"Date: {state['commit_record'].date}\n\n"
                    f"Surrounding commits for context:\n{context_lines}\n\n"
                    f"Diff:\n{diff_text}"
                )
            ),
        ]
    )

    analysis = CommitAnalysis(
        sha=state["sha"],
        era_hint=result.era_hint,
        summary=result.summary,
        narrative_paragraph=result.narrative_paragraph,
        key_changes=result.key_changes,
        additions=additions,
        deletions=deletions,
        date=state["commit_record"].date,
        message=state["commit_record"].message,
        author=state["commit_record"].author,
        tags=state["commit_record"].tags,
    )
    return {"commit_analyses": [analysis]}


# ---------------------------------------------------------------------------
# Node 5: cluster_into_eras
# ---------------------------------------------------------------------------


def cluster_into_eras(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    analyses = state["commit_analyses"]
    # Sort chronologically by date
    analyses_sorted = sorted(analyses, key=lambda a: a.date)

    summaries = "\n".join(
        f"- {a.sha[:8]} ({a.date}): era_hint={a.era_hint!r}, summary={a.summary}"
        for a in analyses_sorted
    )

    llm = _build_llm()
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


# ---------------------------------------------------------------------------
# Node 6: synthesize_outline (deterministic — no LLM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Node 7: expand_to_narrative
# ---------------------------------------------------------------------------


def expand_to_narrative(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    meta = state["repo_metadata"]
    eras = state["eras"]
    analyses_by_sha: dict[str, CommitAnalysis] = {a.sha: a for a in state["commit_analyses"]}

    # Build per-era context for the LLM
    era_blocks: list[str] = []
    for era in eras:
        block_lines = [f"ERA: {era.title} ({era.start_date} – {era.end_date})"]
        block_lines.append(f"Description: {era.description}")
        for sha in era.commit_shas:
            a = analyses_by_sha.get(sha)
            if not a:
                continue
            commit_url = f"https://github.com/{meta.full_name}/commit/{sha}"
            block_lines.append(f"  [{sha[:8]}]({commit_url}): {a.narrative_paragraph}")
        era_blocks.append("\n".join(block_lines))

    all_eras_text = "\n\n".join(era_blocks)

    llm = _build_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a software historian writing a compelling narrative history of a "
                    "GitHub repository. Write in past tense with an authoritative yet engaging "
                    "voice. Structure your narrative around the provided eras. Include an "
                    "introduction paragraph and a conclusion paragraph. "
                    f"Target length: {MIN_NARRATIVE_WORDS}-{MAX_NARRATIVE_WORDS} words. "
                    "CRITICAL: Cite commits inline using Markdown links in the format "
                    "[[sha_prefix]](github_commit_url) — e.g. "
                    "[[a1b2c3d4]](https://github.com/owner/repo/commit/full_sha). "
                    "Include causal analysis — explain WHY changes happened and how they "
                    "connected to each other. Every era must appear as a section."
                )
            ),
            HumanMessage(
                content=(
                    f"Repository: {meta.full_name}\n"
                    f"Description: {meta.description}\n"
                    f"Language: {meta.language} | Stars: {meta.stars}\n\n"
                    f"Era data with commit narratives:\n\n{all_eras_text}"
                )
            ),
        ]
    )

    narrative = response.text

    return {"narrative": narrative}
