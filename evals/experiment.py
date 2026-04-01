"""LangSmith experiment runner for repo-historian evals."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from repo_historian.run import _next_run_prefix

from .evaluators import build_narrative_evaluators, build_step_evaluators
from .pipeline import run_pipeline
from .types import EvalConfig, ReferenceExpectations, TargetOutput

LANGSMITH_MAX_COMMENT_BYTES = 10240


def _truncate_comment(result: dict) -> dict:
    """Truncate feedback comment to fit LangSmith's 10240-byte limit."""
    comment = result.get("comment")
    if comment and len(comment.encode("utf-8")) > LANGSMITH_MAX_COMMENT_BYTES:
        # Truncate to fit; leave room for the ellipsis suffix
        limit = LANGSMITH_MAX_COMMENT_BYTES - 20
        encoded = comment.encode("utf-8")[:limit]
        result["comment"] = encoded.decode("utf-8", errors="ignore") + "\n… [truncated]"
    return result


NARRATIVE_TASK = (
    "Given structured diff analyses of a GitHub repository's commit history "
    "(as JSON), write a narrative history that preserves interesting technical "
    "insights, covers all eras of development, and cites changes with inline "
    "Markdown links to GitHub compare views."
)


# ---------------------------------------------------------------------------
# JSON views for evaluator inputs
# ---------------------------------------------------------------------------


def _analyses_from_raw(raw_data: dict) -> list[dict]:
    return raw_data.get("diff_analyses") or raw_data.get("merged_analyses", [])


def _triage_context_json(analyses: list[dict]) -> str:
    return json.dumps(
        [
            {
                "label": a["label"],
                "key_changes": a["key_changes"],
                "start_date": a["start_date"],
                "end_date": a["end_date"],
                "commit_count": a["commit_count"],
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
        {"key_changes": a["key_changes"]},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Evaluator kwargs by name
# ---------------------------------------------------------------------------


def _narrative_eval_kwargs(name: str, output: TargetOutput, ref: ReferenceExpectations) -> dict:
    """Return the kwargs for a specific narrative evaluator."""
    narrative = output.narrative
    raw_json = json.dumps(output.raw_data, indent=2)

    configs = {
        "hallucination": dict(
            inputs=NARRATIVE_TASK,
            outputs=narrative,
            context=raw_json,
            reference_outputs=ref.expected_narrative_themes,
        ),
        "correctness": dict(inputs=NARRATIVE_TASK, outputs=narrative, reference_outputs=raw_json),
        "insight_preservation": dict(
            inputs=raw_json,
            outputs=narrative,
            reference_outputs=(
                ref.expected_narrative_themes + "\n\n" + ref.narrative_quality_expectations
            ),
        ),
        "completeness": dict(
            inputs=raw_json,
            outputs=narrative,
            reference_outputs=ref.expected_inflection_points,
        ),
        "cross_repo": dict(
            inputs=raw_json,
            outputs=narrative,
            reference_outputs=ref.expected_cross_repo_connections,
        ),
    }
    return configs[name]


# ---------------------------------------------------------------------------
# Build evaluate()-compatible wrappers
# ---------------------------------------------------------------------------


def _build_evaluators(config: EvalConfig, ref: ReferenceExpectations) -> list:
    """Build evaluate()-compatible evaluator wrappers."""
    wrappers: list = []

    if config.eval_scope in ("all", "narrative"):
        narrative_evals = build_narrative_evaluators(config.judge_model)

        def _make_wrapper(name, evaluator):
            def wrapper(inputs: dict, outputs: dict) -> dict:
                output = TargetOutput(narrative=outputs["narrative"], raw_data=outputs["raw_data"])
                return _truncate_comment(evaluator(**_narrative_eval_kwargs(name, output, ref)))

            wrapper.__name__ = f"eval_{name}"
            return wrapper

        wrappers.extend(_make_wrapper(name, ev) for name, ev in narrative_evals.items())

    if config.eval_scope in ("all", "steps"):
        step_evals = build_step_evaluators(config.judge_model)

        def eval_triage_quality(inputs: dict, outputs: dict) -> dict:
            output = TargetOutput(narrative=outputs["narrative"], raw_data=outputs["raw_data"])
            analyses = _analyses_from_raw(output.raw_data)
            return _truncate_comment(
                step_evals["triage_quality"](
                    inputs=_triage_context_json(analyses),
                    outputs=_triage_boundaries_json(analyses),
                    reference_outputs=ref.expected_inflection_points,
                )
            )

        def eval_analysis_quality(inputs: dict, outputs: dict) -> dict:
            output = TargetOutput(narrative=outputs["narrative"], raw_data=outputs["raw_data"])
            analyses = _analyses_from_raw(output.raw_data)

            def _judge_one(a):
                return step_evals["analysis_quality"](
                    inputs=_analysis_input_json(a),
                    outputs=_analysis_output_json(a),
                    reference_outputs=ref.analysis_quality_expectations,
                )

            with ThreadPoolExecutor(max_workers=10) as pool:
                results = list(pool.map(_judge_one, analyses))

            scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
            avg = sum(scores) / len(scores) if scores else 0.0
            per_era = "\n".join(
                f"  {a['label']}: {r.get('score', 'ERR')}" for a, r in zip(analyses, results)
            )
            return _truncate_comment(
                {
                    "key": "analysis_quality",
                    "score": avg,
                    "comment": f"Average of {len(scores)} analyses:\n{per_era}",
                }
            )

        wrappers.extend([eval_triage_quality, eval_analysis_quality])

    return wrappers


# ---------------------------------------------------------------------------
# Dataset sync + experiment entry point
# ---------------------------------------------------------------------------


def ensure_dataset(client, dataset_def: dict):
    """Create the LangSmith dataset + examples if needed."""
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


def run_experiment(dataset_def: dict, config: EvalConfig) -> None:
    """Run pipeline as a LangSmith experiment with evaluators."""
    from langsmith import Client
    from langsmith.evaluation import evaluate

    from repo_historian.config import MODEL_NAME, NARRATIVE_MODEL_NAME, VERSION

    client = Client()
    name = dataset_def["name"]
    ensure_dataset(client, dataset_def)

    ref = ReferenceExpectations.from_dict(dataset_def["examples"][0].get("reference_outputs"))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    run_prefix = _next_run_prefix(out_dir, "eval")

    def target(inputs: dict) -> dict:
        raw_data, narrative, slug = run_pipeline(inputs["repo_urls"], name=name, style=config.style)
        (out_dir / f"{run_prefix}_raw.json").write_text(
            json.dumps(raw_data, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / f"{run_prefix}_narrative.md").write_text(narrative, encoding="utf-8")
        output = TargetOutput(narrative=narrative, raw_data=raw_data)
        return asdict(output)

    evaluators = _build_evaluators(config, ref)

    print(f"\nStarting LangSmith experiment on dataset '{name}'")
    print(f"Judge model: {config.judge_model}")

    evaluate(
        target,
        data=name,
        evaluators=evaluators,
        experiment_prefix=config.experiment_prefix,
        description=f"repo-historian eval: {name}",
        metadata={
            "judge_model": config.judge_model,
            "model_name": MODEL_NAME,
            "narrative_model_name": NARRATIVE_MODEL_NAME,
            "app_version": VERSION,
        },
    )

    print(f"\nExperiment complete. View results in LangSmith under dataset: '{name}'")
