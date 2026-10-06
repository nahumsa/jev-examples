from pathlib import Path

from pydantic_evals import Dataset

from app.problems import BY_ID

from .evaluators import ComparisonImprovement, OutputContract, RubricScoreBands
from .models import EvalInput, EvalOutput, Expectations

CASES_PATH = Path(__file__).with_name("cases.json")
ProofDataset = Dataset[EvalInput, EvalOutput, Expectations]


def load_dataset(path: Path = CASES_PATH) -> ProofDataset:
    dataset = ProofDataset.from_file(path)
    names = set()
    for case in dataset.cases:
        if not case.name or case.name in names:
            raise ValueError("Cases must have unique, non-empty names")
        names.add(case.name)
        question = case.inputs.submission
        if question.problem_id is not None and question.problem_id not in BY_ID:
            raise ValueError(f"Unknown problem in {case.name}")
        expected = case.metadata
        if expected is None:
            raise ValueError(f"Missing labels in {case.name}")
        paired = case.inputs.second_proof is not None
        has_pair_labels = (
            bool(expected.answer_b)
            or expected.minimum_total_improvement is not None
            or expected.minimum_logic_improvement is not None
        )
        if has_pair_labels and not paired:
            raise ValueError(f"Comparison labels require a second proof in {case.name}")
        if paired and not expected.answer_b:
            raise ValueError(f"Comparison cases need labels for both answers in {case.name}")
    dataset.add_evaluator(OutputContract())
    dataset.add_evaluator(RubricScoreBands())
    dataset.add_evaluator(ComparisonImprovement())
    return dataset
