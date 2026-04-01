"""CLI entry point for repo-historian evals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import multi_slug, slugify
from .types import EvalConfig

DEFAULT_JUDGE_MODEL = "openai:gpt-5.4"
DEFAULT_DATASET = "langchain_langgraph"
DATASETS_DIR = Path(__file__).parent / "datasets"


def load_dataset_definition(name: str) -> dict:
    """Load a dataset definition JSON from evals/datasets/."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        print(f"Error: dataset '{name}' not found at {path}", file=sys.stderr)
        available = ", ".join(p.stem for p in DATASETS_DIR.glob("*.json"))
        print(f"Available: {available}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run repo-historian pipeline + LLM-as-judge evals (LangSmith)",
        epilog=(
            "Examples:\n"
            "  uv run python -m evals\n"
            "  uv run python -m evals --repos https://github.com/owner/repo\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset definition name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        metavar="URL",
        help="Repo URLs (alternative to --dataset for ad-hoc runs)",
    )
    parser.add_argument("--style", help="Narrative style hint")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"LLM judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--eval",
        choices=["all", "narrative", "steps"],
        default="all",
        help="Which evals to run (default: all)",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="repo-historian",
        help="LangSmith experiment prefix (default: repo-historian)",
    )
    args = parser.parse_args(argv)

    if args.repos:
        slug = multi_slug(args.repos) if len(args.repos) > 1 else slugify(args.repos[0])
        dataset_def = {
            "name": slug,
            "description": "Ad-hoc evaluation",
            "examples": [{"inputs": {"repo_urls": args.repos}}],
        }
    else:
        dataset_def = load_dataset_definition(args.dataset)

    config = EvalConfig(
        judge_model=args.judge_model,
        eval_scope=args.eval,
        style=args.style,
        experiment_prefix=args.experiment_prefix,
    )

    from .experiment import run_experiment

    run_experiment(dataset_def, config)
