"""CLI runner: run repo-historian pipeline + LLM-as-judge evals.

Pipeline mode (run pipeline + evaluate, tracked as LangSmith experiment):
    uv run python -m evals --dataset langchain_langgraph
    uv run python -m evals --repos https://github.com/owner/repo

Artifact mode (evaluate saved outputs locally, no experiment tracking):
    uv run python -m evals output/foo_raw.json
    uv run python -m evals output/foo_raw.json --dataset langchain_langgraph
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .evaluators import build_narrative_evaluators, build_step_evaluators

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "openai:gpt-5.4-mini"
DATASETS_DIR = Path(__file__).parent / "datasets"

NARRATIVE_TASK = (
    "Given structured diff analyses of a GitHub repository's commit history "
    "(as JSON), write a narrative history that preserves interesting technical "
    "insights, covers all eras of development, and cites changes with inline "
    "Markdown links to GitHub compare views."
)

NO_REFERENCE = "No reference expectations provided."


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset_definition(name: str) -> dict:
    """Load a dataset definition JSON from evals/datasets/."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        print(f"Error: dataset '{name}' not found at {path}", file=sys.stderr)
        available = ", ".join(p.stem for p in DATASETS_DIR.glob("*.json"))
        print(f"Available: {available}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def _ref(reference_outputs: dict | None, key: str) -> str:
    """Extract a reference expectation as a JSON string, or a fallback."""
    if not reference_outputs or key not in reference_outputs:
        return NO_REFERENCE
    val = reference_outputs[key]
    if isinstance(val, str):
        return val
    return json.dumps(val, indent=2)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _slugify(repo_url: str) -> str:
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo_url)
    if not match:
        return "unknown_repo"
    return f"{match.group(1)}_{match.group(2)}"


def _multi_slug(repo_urls: list[str]) -> str:
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
    slug = name or (_multi_slug(repo_urls) if is_multi else _slugify(repo_urls[0]))

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


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _run_eval(evaluator, **kwargs):
    """Run a single evaluator, catching errors gracefully."""
    try:
        return evaluator(**kwargs)
    except Exception as e:
        return {"score": None, "comment": f"Error: {e}"}


def _fmt_score(score) -> str:
    if score is None:
        return " ERR"
    if isinstance(score, float):
        return f"{score:.2f}"
    return str(score)


def _analyses_from_raw(raw_data: dict) -> list[dict]:
    return raw_data.get("diff_analyses") or raw_data.get("merged_analyses", [])


def _triage_context_json(analyses: list[dict]) -> str:
    return json.dumps(
        [
            {
                "label": a["label"],
                "era_hint": a["era_hint"],
                "summary": a["summary"],
                "start_date": a["start_date"],
                "end_date": a["end_date"],
                "commit_count": a["commit_count"],
                "from_message": a.get("from_message", ""),
                "to_message": a.get("to_message", ""),
            }
            for a in analyses
        ],
        indent=2,
    )


def _triage_boundaries_json(analyses: list[dict]) -> str:
    return json.dumps(
        [
            {
                "from_sha": a["from_sha"][:8],
                "to_sha": a["to_sha"][:8],
                "label": a["label"],
                "start_date": a["start_date"],
                "end_date": a["end_date"],
                "commit_count": a["commit_count"],
            }
            for a in analyses
        ],
        indent=2,
    )


def _analysis_input_json(a: dict) -> str:
    return json.dumps(
        {
            "from_sha": a["from_sha"][:8],
            "to_sha": a["to_sha"][:8],
            "label": a["label"],
            "from_message": a.get("from_message", ""),
            "to_message": a.get("to_message", ""),
            "start_date": a["start_date"],
            "end_date": a["end_date"],
            "commit_count": a["commit_count"],
            "additions": a["additions"],
            "deletions": a["deletions"],
        },
        indent=2,
    )


def _analysis_output_json(a: dict) -> str:
    return json.dumps(
        {
            "era_hint": a["era_hint"],
            "summary": a["summary"],
            "narrative_paragraph": a["narrative_paragraph"],
            "key_changes": a["key_changes"],
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# LangSmith experiment mode (pipeline mode)
# ---------------------------------------------------------------------------


def _ensure_dataset(client, dataset_def: dict):
    """Create the LangSmith dataset + examples from definition if needed."""
    name = dataset_def["name"]
    try:
        dataset = client.read_dataset(dataset_name=name)
    except Exception:
        dataset = client.create_dataset(name, description=dataset_def.get("description", ""))
        for ex in dataset_def["examples"]:
            client.create_example(
                inputs=ex["inputs"],
                outputs=ex.get("reference_outputs"),
                dataset_id=dataset.id,
            )
    return dataset


def _build_langsmith_evaluators(
    judge_model: str,
    eval_scope: str,
    ref: dict | None,
) -> list:
    """Build evaluate()-compatible wrappers that inject reference expectations."""
    wrappers: list = []

    if eval_scope in ("all", "narrative"):
        narrative_evals = build_narrative_evaluators(judge_model)

        def _narrative_kwargs(outputs: dict) -> dict[str, dict]:
            raw_json = json.dumps(outputs["raw_data"], indent=2)
            narrative = outputs["narrative"]
            return {
                "hallucination": dict(
                    inputs=NARRATIVE_TASK,
                    outputs=narrative,
                    context=raw_json,
                ),
                "conciseness": dict(inputs=NARRATIVE_TASK, outputs=narrative),
                "correctness": dict(
                    inputs=NARRATIVE_TASK,
                    outputs=narrative,
                    reference_outputs=raw_json,
                ),
                "insight_preservation": dict(
                    inputs=raw_json,
                    outputs=narrative,
                    reference_outputs=_ref(ref, "expected_narrative_themes")
                    + "\n\n"
                    + _ref(ref, "narrative_quality_expectations"),
                ),
                "completeness": dict(
                    inputs=raw_json,
                    outputs=narrative,
                    reference_outputs=_ref(ref, "expected_inflection_points"),
                ),
                "cross_repo": dict(
                    inputs=raw_json,
                    outputs=narrative,
                    reference_outputs=_ref(ref, "expected_cross_repo_connections"),
                ),
            }

        def _make_narrative_wrapper(name):
            def wrapper(inputs: dict, outputs: dict) -> dict:
                kwargs = _narrative_kwargs(outputs)[name]
                return narrative_evals[name](**kwargs)

            wrapper.__name__ = f"eval_{name}"
            return wrapper

        wrappers.extend(_make_narrative_wrapper(name) for name in narrative_evals)

    if eval_scope in ("all", "steps"):
        step_evals = build_step_evaluators(judge_model)

        def eval_triage_quality(inputs: dict, outputs: dict) -> dict:
            analyses = _analyses_from_raw(outputs["raw_data"])
            return step_evals["triage_quality"](
                inputs=_triage_context_json(analyses),
                outputs=_triage_boundaries_json(analyses),
                reference_outputs=_ref(ref, "expected_inflection_points"),
            )

        def eval_analysis_quality(inputs: dict, outputs: dict) -> dict:
            analyses = _analyses_from_raw(outputs["raw_data"])
            aq_ref = _ref(ref, "analysis_quality_expectations")

            def _judge_one(a):
                return _run_eval(
                    step_evals["analysis_quality"],
                    inputs=_analysis_input_json(a),
                    outputs=_analysis_output_json(a),
                    reference_outputs=aq_ref,
                )

            with ThreadPoolExecutor(max_workers=10) as pool:
                results = list(pool.map(_judge_one, analyses))

            scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
            avg = sum(scores) / len(scores) if scores else 0.0
            per_era = "\n".join(
                f"  {_fmt_score(r.get('score'))}  {a['era_hint']}"
                for a, r in zip(analyses, results)
            )
            return {
                "key": "analysis_quality",
                "score": avg,
                "comment": f"Average of {len(scores)} analyses:\n{per_era}",
            }

        wrappers.extend([eval_triage_quality, eval_analysis_quality])

    return wrappers


def run_langsmith_experiment(
    dataset_def: dict,
    style: str | None,
    judge_model: str,
    eval_scope: str,
) -> None:
    """Run pipeline as a LangSmith experiment with evaluators."""
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    name = dataset_def["name"]

    _ensure_dataset(client, dataset_def)

    # Use the first example's reference_outputs for evaluator injection
    ref = dataset_def["examples"][0].get("reference_outputs")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    def target(inputs: dict) -> dict:
        raw_data, narrative, slug = run_pipeline(inputs["repo_urls"], name=name, style=style)
        (out_dir / f"{slug}_raw.json").write_text(
            json.dumps(raw_data, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / f"{slug}_narrative.md").write_text(narrative, encoding="utf-8")
        return {"narrative": narrative, "raw_data": raw_data}

    evaluators = _build_langsmith_evaluators(judge_model, eval_scope, ref)

    from repo_historian.config import MODEL_NAME, NARRATIVE_MODEL_NAME, VERSION

    print(f"\nStarting LangSmith experiment on dataset '{name}'")
    print(f"Judge model: {judge_model}")

    results = evaluate(
        target,
        data=name,
        evaluators=evaluators,
        experiment_prefix="repo-historian",
        description=f"repo-historian eval: {name}",
        metadata={
            "judge_model": judge_model,
            "model_name": MODEL_NAME,
            "narrative_model_name": NARRATIVE_MODEL_NAME,
            "app_version": VERSION,
        },
    )

    # Print results locally
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS  (higher is better, 0.0-1.0)")
    print("=" * 70)
    for row in results:
        eval_results = row["evaluation_results"]["results"]
        for er in eval_results:
            key = er.key if hasattr(er, "key") else er.get("key", "?")
            score = er.score if hasattr(er, "score") else er.get("score")
            comment = er.comment if hasattr(er, "comment") else er.get("comment", "")
            print(f"\n  {key:>22}: {_fmt_score(score)}")
            if comment:
                for line in str(comment).strip().splitlines()[:4]:
                    print(f"  {'':>22}  {line.strip()}")

    print(f"\nView experiment in LangSmith under dataset: '{name}'")


# ---------------------------------------------------------------------------
# Artifact-only evals (no pipeline run, no LangSmith experiment)
# ---------------------------------------------------------------------------


def run_narrative_evals(
    raw_data: dict, narrative: str, judge_model: str, ref: dict | None = None
) -> dict:
    """Run all narrative-step evaluators in parallel."""
    evaluators = build_narrative_evaluators(judge_model)
    raw_json_str = json.dumps(raw_data, indent=2)

    calls = {
        "hallucination": dict(inputs=NARRATIVE_TASK, outputs=narrative, context=raw_json_str),
        "conciseness": dict(inputs=NARRATIVE_TASK, outputs=narrative),
        "correctness": dict(
            inputs=NARRATIVE_TASK,
            outputs=narrative,
            reference_outputs=raw_json_str,
        ),
        "insight_preservation": dict(
            inputs=raw_json_str,
            outputs=narrative,
            reference_outputs=_ref(ref, "expected_narrative_themes")
            + "\n\n"
            + _ref(ref, "narrative_quality_expectations"),
        ),
        "completeness": dict(
            inputs=raw_json_str,
            outputs=narrative,
            reference_outputs=_ref(ref, "expected_inflection_points"),
        ),
        "cross_repo": dict(
            inputs=raw_json_str,
            outputs=narrative,
            reference_outputs=_ref(ref, "expected_cross_repo_connections"),
        ),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            name: pool.submit(_run_eval, evaluators[name], **kwargs)
            for name, kwargs in calls.items()
        }
        for name, future in futures.items():
            results[name] = future.result()

    return results


def run_step_evals(raw_data: dict, judge_model: str, ref: dict | None = None) -> dict:
    """Run triage-quality and per-analysis-quality evaluators."""
    evaluators = build_step_evaluators(judge_model)
    analyses = _analyses_from_raw(raw_data)
    aq_ref = _ref(ref, "analysis_quality_expectations")

    def _eval_one_analysis(a):
        result = _run_eval(
            evaluators["analysis_quality"],
            inputs=_analysis_input_json(a),
            outputs=_analysis_output_json(a),
            reference_outputs=aq_ref,
        )
        return {"era": a["era_hint"], "label": a["label"], **result}

    with ThreadPoolExecutor(max_workers=10) as pool:
        triage_future = pool.submit(
            _run_eval,
            evaluators["triage_quality"],
            inputs=_triage_context_json(analyses),
            outputs=_triage_boundaries_json(analyses),
            reference_outputs=_ref(ref, "expected_inflection_points"),
        )
        analysis_futures = [pool.submit(_eval_one_analysis, a) for a in analyses]

        triage_result = triage_future.result()
        analysis_results = [f.result() for f in analysis_futures]

    return {"triage_quality": triage_result, "analysis_quality": analysis_results}


def print_results(narrative_results: dict, step_results: dict) -> None:
    if narrative_results:
        print("\n" + "=" * 70)
        print("NARRATIVE EVALUATION RESULTS  (higher is better, 0.0-1.0)")
        print("=" * 70)
        for name, result in narrative_results.items():
            print(f"\n  {name:>22}: {_fmt_score(result.get('score'))}")
            comment = result.get("comment", "")
            if comment:
                for line in comment.strip().splitlines()[:4]:
                    print(f"  {'':>22}  {line.strip()}")
                if len(comment.strip().splitlines()) > 4:
                    print(f"  {'':>22}  ...")

    if step_results:
        print("\n" + "=" * 70)
        print("STEP EVALUATION RESULTS  (higher is better, 0.0-1.0)")
        print("=" * 70)

        triage = step_results["triage_quality"]
        print(f"\n  {'triage_quality':>22}: {_fmt_score(triage.get('score'))}")
        comment = triage.get("comment", "")
        if comment:
            for line in comment.strip().splitlines()[:4]:
                print(f"  {'':>22}  {line.strip()}")

        print(f"\n  {'analysis_quality':>22}:")
        scores = []
        for ar in step_results["analysis_quality"]:
            s = ar.get("score")
            if isinstance(s, (int, float)):
                scores.append(s)
            label = ar.get("label", ar.get("era", ""))[:55]
            print(f"  {'':>22}  {_fmt_score(s)}  {label}")
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {'':>22}  ----")
            print(f"  {'':>22}  {avg:.2f}  (average)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run repo-historian pipeline + LLM-as-judge evals",
        epilog=(
            "Examples:\n"
            "  uv run python -m evals --dataset langchain_langgraph\n"
            "  uv run python -m evals --repos https://github.com/owner/repo\n"
            "  uv run python -m evals output/foo_raw.json\n"
            "  uv run python -m evals output/foo_raw.json"
            " --dataset langchain_langgraph\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline mode
    parser.add_argument(
        "--dataset",
        help="Dataset definition name (from evals/datasets/*.json)",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        metavar="URL",
        help="Repo URLs (alternative to --dataset for ad-hoc runs)",
    )
    parser.add_argument("--style", help="Narrative style hint")

    # Artifact mode
    parser.add_argument(
        "raw_json",
        nargs="?",
        help="Path to *_raw.json file (artifact mode)",
    )
    parser.add_argument(
        "--narrative",
        help="Path to *_narrative.md (auto-detected if omitted)",
    )

    # Shared
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
    args = parser.parse_args(argv)

    has_pipeline = args.dataset or args.repos
    if not has_pipeline and not args.raw_json:
        parser.error("Provide --dataset, --repos, or a path to a *_raw.json file")
    if has_pipeline and args.raw_json and not args.dataset:
        parser.error("Cannot combine --repos with a raw_json path")

    # ---- Load dataset definition if specified ----
    dataset_def = None
    ref = None
    if args.dataset:
        dataset_def = load_dataset_definition(args.dataset)
        ref = dataset_def["examples"][0].get("reference_outputs")

    # ---- Pipeline mode: LangSmith experiment ----
    if has_pipeline and not args.raw_json:
        if dataset_def:
            run_langsmith_experiment(
                dataset_def=dataset_def,
                style=args.style,
                judge_model=args.judge_model,
                eval_scope=args.eval,
            )
        else:
            # Ad-hoc --repos without a dataset file
            ad_hoc_def = {
                "name": _multi_slug(args.repos) if len(args.repos) > 1 else _slugify(args.repos[0]),
                "description": "Ad-hoc evaluation",
                "examples": [{"inputs": {"repo_urls": args.repos}}],
            }
            run_langsmith_experiment(
                dataset_def=ad_hoc_def,
                style=args.style,
                judge_model=args.judge_model,
                eval_scope=args.eval,
            )
        return

    # ---- Artifact mode: local eval only ----
    raw_path = Path(args.raw_json)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found", file=sys.stderr)
        sys.exit(1)
    raw_data = json.loads(raw_path.read_text())

    narrative_path = (
        Path(args.narrative)
        if args.narrative
        else raw_path.with_name(raw_path.name.replace("_raw.json", "_narrative.md"))
    )
    if not narrative_path.exists():
        print(f"Error: {narrative_path} not found", file=sys.stderr)
        sys.exit(1)
    narrative = narrative_path.read_text()
    slug = raw_path.stem.replace("_raw", "")

    print(f"Evaluating: {raw_path.name}")
    if dataset_def:
        print(f"Reference expectations: {args.dataset}")
    print(f"Judge model: {args.judge_model}")

    narrative_results = {}
    step_results = {}

    if args.eval in ("all", "narrative"):
        t0 = time.time()
        print("\nRunning narrative evals...", end="", flush=True)
        narrative_results = run_narrative_evals(raw_data, narrative, args.judge_model, ref)
        print(f" done ({time.time() - t0:.1f}s)")

    if args.eval in ("all", "steps"):
        t0 = time.time()
        print("Running step evals...", end="", flush=True)
        step_results = run_step_evals(raw_data, args.judge_model, ref)
        print(f" done ({time.time() - t0:.1f}s)")

    print_results(narrative_results, step_results)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    eval_path = out_dir / f"{slug}_eval.json"
    eval_out = {
        "source": slug,
        "dataset": args.dataset,
        "judge_model": args.judge_model,
        "narrative": narrative_results,
        "steps": step_results,
    }
    eval_path.write_text(json.dumps(eval_out, indent=2, default=str))
    print(f"\nEval results saved to {eval_path}")


if __name__ == "__main__":
    main()
