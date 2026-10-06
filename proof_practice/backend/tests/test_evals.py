import asyncio
from io import StringIO
from unittest.mock import AsyncMock

import pytest
from pydantic_evals import Dataset
from typesafe_sdk import ScoreAnswer, SystemOneResponse, Usage

from app.grading import RUBRIC
from app.main import Settings
from app.problems import BY_ID
from evals.__main__ import REPORT_ADAPTER, main, parser, report_passed, run_live
from evals.dataset import load_dataset
from evals.models import EvalInput, EvalOutput, Expectations
from evals.task import make_task


def response_for(scores):
    answers = {}
    for key, rubric in RUBRIC.items():
        score = scores.get(key, 1.9)
        probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        if score <= 1:
            probabilities.update({0: 1 - score, 1: score})
        else:
            probabilities.update({1: 2 - score, 2: score - 1})
        answers[key] = ScoreAnswer(
            score=score,
            confidence=0.5,
            probabilities=probabilities,
            legend=dict(enumerate(rubric["criteria"])),
        )
    return SystemOneResponse(
        model="jev-offline-stub",
        usage=Usage(input_tokens=100, output_tokens=20),
        answers=answers,
    )


def labeled_stub(dataset):
    """Synthetic provider outputs test the harness, NOT Jev quality."""
    responses = {}
    for case in dataset.cases:
        question = case.inputs.submission
        statement = question.custom_statement or BY_ID[question.problem_id].statement
        for proof, bands in [
            (question.proof, case.metadata.answer_a),
            (case.inputs.second_proof, case.metadata.answer_b),
        ]:
            if proof is not None:
                scores = {key: (band.minimum + band.maximum) / 2 for key, band in bands.items()}
                responses[statement, proof] = response_for(scores)

    async def system_one(state, questions):
        assert set(state) == {"task", "problem", "reference_proof", "student_proof"}
        assert set(questions) == set(RUBRIC)
        return responses[state["problem"], state["student_proof"]]

    return AsyncMock(system_one=AsyncMock(side_effect=system_one))


def test_dataset_coverage():
    dataset = load_dataset()
    assert len(dataset.cases) == 17
    assert {case.inputs.submission.problem_id for case in dataset.cases} >= set(BY_ID)
    assert sum(case.inputs.second_proof is not None for case in dataset.cases) == 2
    assert {case.metadata.category for case in dataset.cases} >= {
        "correct",
        "alternative-method",
        "examples-only",
        "circular",
        "algebra-error",
        "missing-case",
        "missing-base",
        "custom-correct",
        "false-claim",
        "prompt-injection",
        "comparison",
    }


def test_full_harness_with_synthetic_provider():
    dataset = load_dataset()
    client = labeled_stub(dataset)
    report = asyncio.run(dataset.evaluate(make_task(client), progress=False, max_concurrency=2))
    assert report_passed(report)
    assert client.system_one.await_count == 19
    assert len(report.cases) == 17
    for case in report.cases:
        calls = 2 if case.inputs.second_proof else 1
        assert case.metrics["jev_calls"] == calls
        assert case.metrics["input_tokens"] == calls * 100
        assert case.attributes["resolved_model"] == "jev-offline-stub"
    encoded = REPORT_ADAPTER.dump_json(report)
    decoded = REPORT_ADAPTER.validate_json(encoded)
    assert report_passed(decoded)


def test_repeat_keeps_case_metrics_isolated():
    dataset = load_dataset()
    dataset.cases[:] = [dataset.cases[0], dataset.cases[-2]]
    client = labeled_stub(dataset)
    report = asyncio.run(
        dataset.evaluate(make_task(client), progress=False, max_concurrency=2, repeat=3)
    )
    assert report_passed(report)
    assert len(report.cases) == 6
    assert client.system_one.await_count == 9
    assert sum(case.metrics["jev_calls"] for case in report.cases) == 9


def test_overgrading_an_invalid_proof_fails():
    dataset = load_dataset()
    dataset.cases[:] = [case for case in dataset.cases if case.name == "examples-are-not-a-proof"]
    client = AsyncMock(system_one=AsyncMock(return_value=response_for({})))
    report = asyncio.run(dataset.evaluate(make_task(client), progress=False))
    assert not report_passed(report)
    case = report.cases[0]
    assert case.assertions["a_complete_rubric"].value
    assert not case.assertions["a_logical_reasoning_in_band"].value
    assert case.scores["mean_band_violation"].value > 0


def test_comparison_does_not_reward_worse_answer():
    dataset = load_dataset()
    dataset.cases[:] = [
        case for case in dataset.cases if case.name == "comparison-revision-improves"
    ]
    client = AsyncMock(system_one=AsyncMock(return_value=response_for({})))
    report = asyncio.run(dataset.evaluate(make_task(client), progress=False))
    assert not report_passed(report)
    assert not report.cases[0].assertions["total_ranking"].value
    assert not report.cases[0].assertions["logic_ranking"].value


def test_inconsistent_total_is_caught():
    dataset = load_dataset()
    dataset.cases[:] = dataset.cases[:1]
    client = labeled_stub(dataset)
    task = make_task(client)

    async def broken_task(inputs):
        output = await task(inputs)
        return output.model_copy(
            update={"answer_a": output.answer_a.model_copy(update={"total": 0})}
        )

    report = asyncio.run(dataset.evaluate(broken_task, progress=False))
    assert not report_passed(report)
    assert not report.cases[0].assertions["a_consistent_total"].value


def test_task_failure_is_not_a_quality_pass():
    dataset = load_dataset()
    dataset.cases[:] = dataset.cases[:1]

    async def fail(inputs):
        raise RuntimeError("offline failure")

    report = asyncio.run(dataset.evaluate(fail, progress=False))
    assert report.failures
    assert not report_passed(report)


def test_list_never_constructs_a_provider(monkeypatch, capsys):
    def forbidden(**kwargs):
        pytest.fail("No-cost preview constructed a client")

    monkeypatch.setattr("evals.__main__.AsyncTypeSafeClient", forbidden)
    assert main(["--list"]) == 0
    assert "19 Jev calls" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv", [[], ["--allow-paid"], ["--list", "--repeat", "0"], ["--list", "--case", "unknown"]]
)
def test_cli_rejects_invalid_or_unauthorized_run(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_cli_refuses_to_overwrite_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--allow-paid", "--output", str(output)])
    assert output.read_text() == "existing"


def test_live_runner_report_with_mocked_provider(monkeypatch):
    dataset = load_dataset()
    dataset.cases[:] = dataset.cases[:1]
    client = labeled_stub(dataset)
    client.__aenter__.return_value = client
    # SDK construction is synchronous, while its context manager and API are async.
    monkeypatch.setattr("evals.__main__.AsyncTypeSafeClient", lambda **kwargs: client)
    settings = Settings(_env_file=None, typesafe_api_key="offline-test-key")
    args = parser().parse_args(["--allow-paid", "--model", "jev-test"])
    output = StringIO()
    assert asyncio.run(run_live(dataset, settings, args, output))
    report = REPORT_ADAPTER.validate_json(output.getvalue())
    assert report.experiment_metadata["requested_model"] == "jev-test"
    assert "dataset_sha256" in report.experiment_metadata
    assert "offline-test-key" not in output.getvalue()
    client.system_one.assert_awaited_once()


def test_loader_rejects_unknown_problem(tmp_path):
    dataset = load_dataset()
    dataset.cases[:] = dataset.cases[:1]
    dataset.cases[0].inputs.submission.problem_id = "missing"
    path = tmp_path / "bad.json"
    # Save plain case data, with no attached evaluator registry required at load time.
    Dataset[EvalInput, EvalOutput, Expectations](
        name="invalid-fixture", cases=dataset.cases
    ).to_file(path)
    with pytest.raises(ValueError, match="Unknown problem"):
        load_dataset(path)
