from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, EvaluatorOutput

from app.grading import RUBRIC, TIPS
from app.models import Evaluation

from .models import EvalInput, EvalOutput, Expectations

Context = EvaluatorContext[EvalInput, EvalOutput, Expectations]


def contract_checks(output: Evaluation) -> dict[str, bool]:
    dimensions = output.dimensions
    return {
        "complete_rubric": len(dimensions) == len(RUBRIC)
        and {d.id for d in dimensions} == set(RUBRIC),
        "consistent_total": abs(output.total - sum(d.score for d in dimensions)) < 1e-6,
        "feedback_matches_rubric": all(
            d.id in RUBRIC
            and d.criteria == RUBRIC[d.id]["criteria"]
            and d.likely_level == max(d.probabilities, key=d.probabilities.__getitem__)
            and d.rubric_feedback == RUBRIC[d.id]["criteria"][d.likely_level]
            and d.revision_tip == TIPS[d.id]
            for d in dimensions
        ),
    }


@dataclass
class OutputContract(Evaluator[EvalInput, EvalOutput, Expectations]):
    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        result: dict[str, bool] = {}
        for label, answer in [("a", ctx.output.answer_a), ("b", ctx.output.answer_b)]:
            if answer is not None:
                result.update({f"{label}_{k}": v for k, v in contract_checks(answer).items()})
                expected_id = ctx.inputs.submission.problem_id or "custom"
                result[f"{label}_question_matches"] = answer.problem_id == expected_id
        paired = ctx.inputs.second_proof is not None
        result["comparison_shape"] = (ctx.output.answer_b is not None) == paired
        result["bounded_jev_calls"] = ctx.metrics.get("jev_calls") == (2 if paired else 1)
        return result


@dataclass
class RubricScoreBands(Evaluator[EvalInput, EvalOutput, Expectations]):
    """Check hand-authored intervals, not exact equality on stochastic scores."""

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        expected = ctx.metadata
        if expected is None:
            return {"labels_present": False}
        results: dict[str, EvaluationReason | float] = {}
        errors = []
        for label, answer, bands in [
            ("a", ctx.output.answer_a, expected.answer_a),
            ("b", ctx.output.answer_b, expected.answer_b),
        ]:
            scores = {} if answer is None else {d.id: d.score for d in answer.dimensions}
            for key, band in bands.items():
                score = scores.get(key)
                passed = score is not None and band.minimum <= score <= band.maximum
                results[f"{label}_{key}_in_band"] = EvaluationReason(
                    value=passed,
                    reason=f"Expected [{band.minimum}, {band.maximum}], got {score}. "
                    f"{expected.rationale}",
                )
                errors.append(
                    2.0 if score is None else max(band.minimum - score, score - band.maximum, 0)
                )
        results["mean_band_violation"] = sum(errors) / len(errors) if errors else 0
        return results


@dataclass
class ComparisonImprovement(Evaluator[EvalInput, EvalOutput, Expectations]):
    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        expected = ctx.metadata
        if expected is None or ctx.inputs.second_proof is None:
            return {}
        if ctx.output.answer_b is None:
            return {"comparison_available": False}
        a, b = ctx.output.answer_a, ctx.output.answer_b
        delta = b.total - a.total
        results: dict[str, EvaluationReason | float] = {"total_improvement": delta}
        if expected.minimum_total_improvement is not None:
            results["total_ranking"] = EvaluationReason(
                value=delta >= expected.minimum_total_improvement,
                reason=f"B − A = {delta:.3f}; require ≥ {expected.minimum_total_improvement}",
            )
        if expected.minimum_logic_improvement is not None:
            a_logic = next(d.score for d in a.dimensions if d.id == "logical_reasoning")
            b_logic = next(d.score for d in b.dimensions if d.id == "logical_reasoning")
            logic_delta = b_logic - a_logic
            results["logic_ranking"] = EvaluationReason(
                value=logic_delta >= expected.minimum_logic_improvement,
                reason=f"Logic B − A = {logic_delta:.3f}; "
                f"require ≥ {expected.minimum_logic_improvement}",
            )
        return results
