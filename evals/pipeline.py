"""Pipeline execution for repo-historian evals."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict

_GITHUB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")


def _parse_github_url(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL."""
    m = _GITHUB_RE.match(url)
    return (m.group(1), m.group(2)) if m else None


def slugify(repo_url: str) -> str:
    parsed = _parse_github_url(repo_url)
    if not parsed:
        return "unknown_repo"
    return f"{parsed[0]}_{parsed[1]}"


def multi_slug(repo_urls: list[str]) -> str:
    orgs: set[str] = set()
    repo_names: list[str] = []
    for url in repo_urls:
        parsed = _parse_github_url(url)
        if parsed:
            orgs.add(parsed[0])
            repo_names.append(parsed[1])
    if len(orgs) == 1:
        return f"{next(iter(orgs))}_ecosystem"
    return "_".join(repo_names[:4])


def run_pipeline(
    repo_urls: list[str], name: str | None = None, style: str | None = None
) -> tuple[dict, str, str]:
    """Run the repo-historian pipeline. Returns (raw_data, narrative, slug)."""
    from repo_historian.config import (
        MODEL_NAME,
        NARRATIVE_MODEL_NAME,
        VERSION,
        detect_provider,
    )

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: GITHUB_TOKEN is required", file=sys.stderr)
        sys.exit(1)

    is_multi = len(repo_urls) > 1
    slug = name or (multi_slug(repo_urls) if is_multi else slugify(repo_urls[0]))

    run_config = {
        "run_name": f"repo-historian-eval:{slug}",
        "tags": ["repo-historian", "eval", detect_provider(MODEL_NAME)],
        "metadata": {
            "repo_urls": repo_urls,
            "model_name": MODEL_NAME,
            "narrative_model_name": NARRATIVE_MODEL_NAME,
            "app_version": VERSION,
        },
        "configurable": {
            "github_token": github_token,
            "style": style,
            "skip_confirmation": True,
        },
    }

    if is_multi:
        from repo_historian.multi_repo_graph import build_multi_repo_graph

        graph = build_multi_repo_graph()
        print(f"Running multi-repo pipeline on {len(repo_urls)} repos...")
        for url in repo_urls:
            print(f"  - {url}")
        result = graph.invoke({"repo_urls": repo_urls}, config=run_config)
        raw_data = {
            "all_repo_metadata": [asdict(m) for m in result["all_repo_metadata"]],
            "merged_analyses": [asdict(a) for a in result["merged_analyses"]],
        }
    else:
        from repo_historian.graph import build_graph

        graph = build_graph()
        print(f"Running single-repo pipeline on {repo_urls[0]}...")
        result = graph.invoke({"repo_url": repo_urls[0]}, config=run_config)
        raw_data = {
            "repo_metadata": asdict(result["repo_metadata"]),
            "diff_analyses": [asdict(a) for a in result["diff_analyses"]],
        }

    return raw_data, result["narrative"], slug
