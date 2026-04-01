"""Typed data structures for eval configuration and dataset references."""

from __future__ import annotations

import json
from dataclasses import dataclass

NO_REFERENCE = "No reference expectations provided."


@dataclass
class ReferenceExpectations:
    """Typed access to reference_outputs from dataset examples."""

    expected_inflection_points: str = NO_REFERENCE
    expected_narrative_themes: str = NO_REFERENCE
    narrative_quality_expectations: str = NO_REFERENCE
    expected_cross_repo_connections: str = NO_REFERENCE
    analysis_quality_expectations: str = NO_REFERENCE

    @classmethod
    def from_dict(cls, d: dict | None) -> ReferenceExpectations:
        """Parse from dataset JSON, JSON-encoding lists and defaulting missing keys."""
        if not d:
            return cls()

        def _coerce(val) -> str:
            if val is None:
                return NO_REFERENCE
            if isinstance(val, str):
                return val
            return json.dumps(val, indent=2)

        return cls(
            expected_inflection_points=_coerce(d.get("expected_inflection_points")),
            expected_narrative_themes=_coerce(d.get("expected_narrative_themes")),
            narrative_quality_expectations=_coerce(d.get("narrative_quality_expectations")),
            expected_cross_repo_connections=_coerce(d.get("expected_cross_repo_connections")),
            analysis_quality_expectations=_coerce(d.get("analysis_quality_expectations")),
        )


@dataclass
class EvalConfig:
    """All settings for an eval run."""

    judge_model: str
    eval_scope: str  # "all" | "narrative" | "steps"
    style: str | None = None
    experiment_prefix: str = "repo-historian"


@dataclass
class TargetOutput:
    """Typed contract between the pipeline target and evaluators."""

    narrative: str
    raw_data: dict
