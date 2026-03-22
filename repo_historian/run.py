"""CLI entry point: python -m repo_historian.run <url> [--batch-size N]."""

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
    DEFAULT_TRIAGE_BATCH_SIZE,
    MODEL_NAME,
    PROVIDER_API_KEY_ENV,
    VERSION,
    detect_provider,
)
from repo_historian.graph import build_graph
from repo_historian.state import TriageConfig


def _slugify(repo_url: str) -> str:
    """Convert 'https://github.com/owner/repo' → 'owner_repo'."""
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo_url)
    if not match:
        return "unknown_repo"
    return f"{match.group(1)}_{match.group(2)}"


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Repo Historian — narrative Git history")
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRIAGE_BATCH_SIZE)
    args = parser.parse_args(argv)

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

    graph = build_graph()

    initial_state = {
        "repo_url": args.url,
        "triage_config": TriageConfig(batch_size=args.batch_size),
    }

    print(f"Processing {args.url} (model={MODEL_NAME}, batch-size={args.batch_size})...")

    slug = _slugify(args.url)
    run_config = {
        "run_name": f"repo-historian:{slug}",
        "tags": ["repo-historian", provider],
        "metadata": {
            "repo_url": args.url,
            "model_name": MODEL_NAME,
            "batch_size": args.batch_size,
            "app_version": VERSION,
        },
        "configurable": {
            "github_token": github_token,
        },
    }

    result = graph.invoke(initial_state, config=run_config)

    # Write outputs
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    outline_path = out_dir / f"{slug}_outline.md"
    narrative_path = out_dir / f"{slug}_narrative.md"
    raw_path = out_dir / f"{slug}_raw.json"

    outline_path.write_text(result["outline"], encoding="utf-8")
    narrative_path.write_text(result["narrative"], encoding="utf-8")

    # Serialize raw data
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


if __name__ == "__main__":
    main()
