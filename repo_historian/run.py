"""CLI entry point: python -m repo_historian <url> or --repos <url1> <url2> ..."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from repo_historian.config import (
    MODEL_NAME,
    PROVIDER_API_KEY_ENV,
    VERSION,
    detect_provider,
)


def _slugify(repo_url: str) -> str:
    """Convert 'https://github.com/owner/repo' → 'owner_repo'."""
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo_url)
    if not match:
        return "unknown_repo"
    return f"{match.group(1)}_{match.group(2)}"


def _multi_repo_slug(repo_urls: list[str]) -> str:
    """Derive a slug for multi-repo output files."""
    orgs: set[str] = set()
    repo_names: list[str] = []
    for url in repo_urls:
        match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
        if match:
            orgs.add(match.group(1))
            repo_names.append(match.group(2))
    if len(orgs) == 1:
        return f"{next(iter(orgs))}_ecosystem"
    return "_".join(repo_names[:4])


def _run_single_repo(args: argparse.Namespace, github_token: str) -> None:
    """Run the original single-repo pipeline."""
    from repo_historian.graph import build_graph

    graph = build_graph()

    initial_state = {
        "repo_url": args.url,
    }

    slug = args.name if args.name else _slugify(args.url)
    run_config = {
        "run_name": f"repo-historian:{slug}",
        "tags": ["repo-historian", detect_provider(MODEL_NAME)],
        "metadata": {
            "repo_url": args.url,
            "model_name": MODEL_NAME,
            "app_version": VERSION,
        },
        "configurable": {
            "github_token": github_token,
        },
    }

    print(f"Processing {args.url} (model={MODEL_NAME})...")
    result = graph.invoke(initial_state, config=run_config)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    outline_path = out_dir / f"{slug}_outline.md"
    narrative_path = out_dir / f"{slug}_narrative.md"
    raw_path = out_dir / f"{slug}_raw.json"

    outline_path.write_text(result["outline"], encoding="utf-8")
    narrative_path.write_text(result["narrative"], encoding="utf-8")

    raw_data = {
        "repo_metadata": asdict(result["repo_metadata"]),
        "diff_analyses": [asdict(a) for a in result["diff_analyses"]],
        "eras": [asdict(e) for e in result["eras"]],
    }
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str), encoding="utf-8")

    print("\nDone! Outputs written to:")
    print(f"  Outline:   {outline_path}")
    print(f"  Narrative: {narrative_path}")
    print(f"  Raw JSON:  {raw_path}")


def _run_multi_repo(args: argparse.Namespace, github_token: str) -> None:
    """Run the multi-repo orchestrator pipeline."""
    from repo_historian.multi_repo_graph import build_multi_repo_graph

    repo_urls = args.repos
    graph = build_multi_repo_graph()

    initial_state = {
        "repo_urls": repo_urls,
    }

    slug = args.name if args.name else _multi_repo_slug(repo_urls)
    provider = detect_provider(MODEL_NAME)
    run_config = {
        "run_name": f"repo-historian-multi:{slug}",
        "tags": ["repo-historian", "multi-repo", provider],
        "metadata": {
            "repo_urls": repo_urls,
            "model_name": MODEL_NAME,
            "app_version": VERSION,
        },
        "configurable": {
            "github_token": github_token,
        },
    }

    print(f"Processing {len(repo_urls)} repositories (model={MODEL_NAME}):")
    for url in repo_urls:
        print(f"  - {url}")

    result = graph.invoke(initial_state, config=run_config)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    outline_path = out_dir / f"{slug}_outline.md"
    narrative_path = out_dir / f"{slug}_narrative.md"
    raw_path = out_dir / f"{slug}_raw.json"

    outline_path.write_text(result["outline"], encoding="utf-8")
    narrative_path.write_text(result["narrative"], encoding="utf-8")

    raw_data = {
        "all_repo_metadata": [asdict(m) for m in result["all_repo_metadata"]],
        "merged_analyses": [asdict(a) for a in result["merged_analyses"]],
        "cross_repo_eras": [asdict(e) for e in result["cross_repo_eras"]],
    }
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str), encoding="utf-8")

    print("\nDone! Outputs written to:")
    print(f"  Outline:   {outline_path}")
    print(f"  Narrative: {narrative_path}")
    print(f"  Raw JSON:  {raw_path}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Repo Historian — narrative Git history")
    parser.add_argument("url", nargs="?", help="GitHub repository URL (single-repo mode)")
    parser.add_argument(
        "--repos",
        nargs="+",
        metavar="URL",
        help="Multiple GitHub repository URLs (multi-repo mode)",
    )
    parser.add_argument(
        "--name",
        help="Name for the run (used in output filenames)",
    )
    args = parser.parse_args(argv)

    if not args.url and not args.repos:
        parser.error("Provide either a single URL or --repos <url1> <url2> ...")
    if args.url and args.repos:
        parser.error("Cannot combine a positional URL with --repos")

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    provider = detect_provider(MODEL_NAME)
    api_key_env = PROVIDER_API_KEY_ENV[provider]
    if not os.environ.get(api_key_env, ""):
        print(
            f"Error: {api_key_env} environment variable is required for model {MODEL_NAME}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.repos:
        _run_multi_repo(args, github_token)
    else:
        _run_single_repo(args, github_token)


if __name__ == "__main__":
    main()
