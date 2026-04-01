"""LLM-as-judge evaluators for repo-historian pipeline stages."""

from __future__ import annotations

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT

INSIGHT_PRESERVATION_PROMPT = """\
You are evaluating whether a narrative history of a software project preserves \
the most interesting and technically specific insights from its source material.

<Inputs>
{inputs}
</Inputs>

<Outputs>
{outputs}
</Outputs>

<Reference expectations>
{reference_outputs}
</Reference expectations>

Evaluate:
1. Are specific technical decisions and patterns from the analyses preserved? \
(e.g. concrete config names, architecture patterns, design tradeoffs)
2. Are "why" explanations retained — not just what changed, but why it mattered?
3. Does the narrative add interpretive depth, or just summarize?
4. Are concrete details kept, or flattened into vague generalities?
5. Does the narrative cover the themes and technical concepts described in the \
reference expectations? These represent what a knowledgeable human considers the \
most important insights this narrative should surface.

Score 1.0 if every interesting technical insight from the source appears in the \
narrative with appropriate depth and the reference expectations are well-covered. \
Score 0.0 if the narrative is purely generic with no preserved specifics."""

COMPLETENESS_PROMPT = """\
You are evaluating whether a narrative history covers all distinct eras from \
the source analyses it was generated from.

<Inputs>
{inputs}
</Inputs>

<Outputs>
{outputs}
</Outputs>

<Reference expectations>
{reference_outputs}
</Reference expectations>

The inputs contain diff_analyses, each representing a distinct era of the \
project's history. The reference expectations list inflection points and themes \
that a knowledgeable human identified as critical. Check:
1. Does every era from the analyses appear in the narrative?
2. Are any eras merged in a way that loses their distinct contributions?
3. Is the chronological flow maintained?
4. Is emphasis proportional to each era's significance?
5. Are the expected inflection points from the reference adequately represented?

Score 1.0 if every era is distinctly represented and expected inflection points \
are covered. Score 0.0 if major eras or expected milestones are missing."""

ANALYSIS_QUALITY_PROMPT = """\
You are evaluating the quality of key_changes bullets produced from analyzing \
a range of code changes in a software project.

<Inputs>
{inputs}
</Inputs>

<Outputs>
{outputs}
</Outputs>

<Quality expectations>
{reference_outputs}
</Quality expectations>

The input is metadata about a range of commits (dates, size). The output is a \
list of key_changes bullets summarizing the most important technical changes. \
The quality expectations describe what depth of insight is expected.

Evaluate:
1. Are the bullets specific and technically precise, or vague?
2. Do they name concrete technologies, patterns, config keys, and tradeoffs?
3. Do they capture motivations and "why" — not just what changed?
4. Are they information-dense and self-contained?
5. Do they meet the quality expectations for technical depth?

Score 1.0 if the bullets are insightful, specific, and capture what matters. \
Score 0.0 if they read like a generic changelog."""

TRIAGE_QUALITY_PROMPT = """\
You are evaluating whether a set of chapter boundaries chosen for a software \
project's narrative history are well-selected.

<Inputs>
{inputs}
</Inputs>

<Outputs>
{outputs}
</Outputs>

<Expected inflection points>
{reference_outputs}
</Expected inflection points>

The inputs contain key changes and context for each chapter of the project's \
history. The outputs show the chosen boundaries. The expected inflection points \
are milestones that a knowledgeable human identified as critical turning points \
this triage should capture. Evaluate:
1. Do the boundaries represent genuine changes in direction, architecture, \
or character?
2. Is the granularity appropriate — not too many trivial splits, not so few \
that distinct phases are merged?
3. Do the labels accurately describe why each boundary is a turning point?
4. How well do the chosen boundaries align with the expected inflection points? \
Are any expected milestones missing or merged away?

Score 1.0 if the boundaries capture all expected inflection points and partition \
the history into meaningful chapters. Score 0.0 if they are arbitrary or miss \
critical turning points."""

CROSS_REPO_PROMPT = """\
You are evaluating whether a multi-repository narrative history successfully \
identifies and explains connections between the repositories.

<Inputs>
{inputs}
</Inputs>

<Outputs>
{outputs}
</Outputs>

<Expected cross-repo connections>
{reference_outputs}
</Expected cross-repo connections>

Evaluate:
1. Does the narrative explain how changes in one repo motivated or enabled \
changes in another?
2. Are the expected cross-repo connections from the reference adequately covered?
3. Does the narrative weave the repos into a coherent ecosystem story, or treat \
them as independent summaries placed side by side?
4. Are temporal correlations between repos noted where relevant?

Score 1.0 if the narrative is a genuinely integrated ecosystem story that covers \
the expected connections. Score 0.0 if the repos are treated independently with \
no cross-repo insight."""


NARRATIVE_EVALUATOR_PROMPTS: dict[str, str] = {
    "hallucination": HALLUCINATION_PROMPT,
    "correctness": CORRECTNESS_PROMPT,
    "insight_preservation": INSIGHT_PRESERVATION_PROMPT,
    "completeness": COMPLETENESS_PROMPT,
    "cross_repo": CROSS_REPO_PROMPT,
}

STEP_EVALUATOR_PROMPTS: dict[str, str] = {
    "analysis_quality": ANALYSIS_QUALITY_PROMPT,
    "triage_quality": TRIAGE_QUALITY_PROMPT,
}


def _build_evaluators(prompts: dict[str, str], judge_model: str) -> dict:
    return {
        name: create_llm_as_judge(
            prompt=prompt,
            model=judge_model,
            continuous=True,
            feedback_key=name,
        )
        for name, prompt in prompts.items()
    }


def build_narrative_evaluators(judge_model: str) -> dict:
    """Build evaluators for the narrative generation step."""
    return _build_evaluators(NARRATIVE_EVALUATOR_PROMPTS, judge_model)


def build_step_evaluators(judge_model: str) -> dict:
    """Build evaluators for individual pipeline steps."""
    return _build_evaluators(STEP_EVALUATOR_PROMPTS, judge_model)
