import asyncio
import json
from collections import Counter
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import infer_model
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from pydantic_ai.usage import RunUsage
from typesafe_sdk import AsyncTypeSafeClient, ChoiceAnswer, NoulAnswer

from sentiment_benchmark.__main__ import main
from sentiment_benchmark.data import (
    SHA256,
    SOURCES,
    Sentence,
    load_dataset,
    parse_sentences,
    select_sentences,
    split_for,
    validation_sentences,
)
from sentiment_benchmark.evaluation import comparison_metrics, metrics, run
from sentiment_benchmark.model import (
    ComparisonPrediction,
    Prediction,
    make_agent,
    predict,
)


@pytest.fixture(autouse=True)
def forbid_unmocked_inference(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("Offline tests must not invoke the real TypeSafe API")

    monkeypatch.setattr(AsyncTypeSafeClient, "system_one", forbidden)


def sample_rows():
    return [
        Sentence(f"{source}:{label}:{i}", source, f"A {label} review {i}", label)
        for source in SOURCES
        for label in ("negative", "positive")
        for i in range(10)
    ]


def validation_fixture():
    rows = []
    for source in SOURCES:
        for label in ("negative", "positive"):
            for i in range(100):
                row = Sentence(
                    f"validation-{source}-{label}-{i}",
                    source,
                    f"{label} example {i}",
                    label,
                )
                if split_for(row) == "validation":
                    rows.append(row)
    return rows


def test_validation_split_is_fixed_disjoint_and_groups_duplicates():
    rows = validation_fixture() + sample_rows()
    validation = validation_sentences(rows)
    development = [row for row in rows if split_for(row) == "development"]
    assert validation and development
    assert {row.id for row in validation}.isdisjoint(row.id for row in development)
    assert {row.id for row in validation} == {
        row.id for row in validation_sentences(list(reversed(rows)))
    }
    duplicates = [
        Sentence("one", "imdb", "GREAT   movie", "positive"),
        Sentence("two", "yelp", " great movie ", "negative"),
        Sentence("three", "amazon_cells", "ＧＲＥＡＴ movie", "positive"),
    ]
    assert len({split_for(row) for row in duplicates}) == 1


@pytest.mark.parametrize("compare_noul", [False, True])
def test_decimal_usage_cost_is_json_safe(tmp_path, compare_noul):
    cost = Decimal("0.00000084")
    output = (
        ComparisonPrediction(sentiment="positive", positive_probability=0.9)
        if compare_noul
        else Prediction(sentiment="positive")
    )
    result = SimpleNamespace(
        output=output,
        usage=RunUsage(input_tokens=20, requests=1, cost=cost),
        response=SimpleNamespace(provider_details={}, model_name="offline-cost-test"),
    )
    agent = SimpleNamespace(run=AsyncMock(return_value=result))
    path = tmp_path / "decimal.jsonl"
    summary = asyncio.run(
        run(
            [Sentence("test", "yelp", "Great!", "positive")],
            output=path,
            manifest={"compare_noul": compare_noul},
            agent=agent,
        )
    )
    assert summary["status"] == "complete"
    record = json.loads(path.read_text())
    assert Decimal(record["usage"]["cost"]) == cost
    assert record["usage"]["input_tokens"] == 20
    assert record["usage"]["requests"] == 1
    if compare_noul:
        assert isinstance(record["noul"]["positive_probability"], float)


def provider(choice="positive"):
    client = AsyncTypeSafeClient(api_key="offline-test-key")
    client.system_one = AsyncMock(
        return_value=SimpleNamespace(
            model="jev-offline-stub",
            request_id="offline-test",
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
            answers={
                "sentiment": ChoiceAnswer(
                    choice=choice,
                    confidence=0.8,
                    probabilities={
                        "negative": 0.1 if choice == "positive" else 0.9,
                        "positive": 0.9 if choice == "positive" else 0.1,
                    },
                )
            },
        )
    )
    return client


def mocked_agent(client, *, compare_noul=False):
    with patch(
        "pydantic_ai.models.infer_model",
        side_effect=lambda model: infer_model(
            model,
            provider_factory=lambda provider: TypeSafeProvider(typesafe_client=client),
        ),
    ):
        return make_agent("jev-1.13.0", compare_noul=compare_noul)


@pytest.mark.parametrize("compare_noul", [False, True])
def test_agent_uses_native_typesafe_model_string(compare_noul):
    with patch("sentiment_benchmark.model.Agent") as constructor:
        assert (
            make_agent("jev-1.13.0", compare_noul=compare_noul)
            is constructor.return_value
        )
    constructor.assert_called_once_with(
        "typesafe:jev-1.13.0",
        output_type=ComparisonPrediction if compare_noul else Prediction,
        retries=0,
        model_settings={"timeout": 30},
    )


def test_parser_preserves_quotes_tabs_unicode_separators():
    rows = parse_sentences('"A\u0085bad\tfilm"  \t0\nGreat!\t1\n', "imdb")
    assert len(rows) == 2
    assert rows[0].text == '"A\u0085bad\tfilm"  '
    assert rows[0].gold == "negative"
    assert rows[1].id == "imdb:2"
    assert rows[1].gold == "positive"


@pytest.mark.parametrize("text", ["No label", "text\t2", "\t1", "text\tpositive"])
def test_parser_rejects_invalid_rows(text):
    with pytest.raises(ValueError, match="Invalid sentence"):
        parse_sentences(text, "yelp")


def test_balanced_reproducible_selection():
    rows = sample_rows()
    selected = select_sentences(rows, source="all", limit=30, seed=42)
    assert len(selected) == len({row.id for row in selected}) == 30
    assert Counter((row.source, row.gold) for row in selected) == {
        (source, label): 5 for source in SOURCES for label in ("negative", "positive")
    }
    assert selected == select_sentences(
        list(reversed(rows)), source="all", limit=30, seed=42
    )
    assert selected != select_sentences(rows, source="all", limit=30, seed=43)


@pytest.mark.parametrize("source,limit", [("all", 1), ("yelp", 3), ("yelp", 1002)])
def test_invalid_sample_size(source, limit):
    with pytest.raises(ValueError):
        select_sentences(sample_rows(), source=source, limit=limit, seed=42)


def test_checksum_rejects_corruption(tmp_path):
    path = tmp_path / "corrupt.zip"
    path.write_bytes(b"not the UCI zip")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path)


def test_prediction_never_sends_gold_or_source():
    client = provider()
    row = Sentence("imdb:1", "imdb", "I liked it.", "positive")
    record = asyncio.run(predict(mocked_agent(client), row))
    assert record["correct"]
    assert record["resolved_model"] == "jev-offline-stub"
    client.system_one.assert_awaited_once()
    state, questions = client.system_one.call_args.args
    assert "I liked it." in str(state)
    assert all(value not in str(state) for value in ("gold", "imdb", "imdb:1"))
    assert set(questions) == {"sentiment"}
    assert set(questions["sentiment"].criteria) == {"negative", "positive"}
    assert "dissatisfaction" in str(questions["sentiment"].criteria["negative"])
    assert record["usage"]["requests"] == 1
    assert record["confidence"] == 0.8
    assert record["probabilities"] == {"negative": 0.1, "positive": 0.9}


@pytest.mark.parametrize("sentiment", ["neutral", "unknown", ""])
def test_output_is_closed_set(sentiment):
    with pytest.raises(ValidationError):
        Prediction(sentiment=sentiment)


def test_pydanticai_typed_output_with_test_model():
    agent = Agent(
        TestModel(custom_output_args={"sentiment": "negative"}), output_type=Prediction
    )
    record = asyncio.run(
        predict(agent, Sentence("yelp:1", "yelp", "Not good.", "negative"))
    )
    assert record["correct"]
    assert record["usage"]["requests"] == 1
    assert record["confidence"] is None


@pytest.mark.parametrize(
    "probability,expected", [(0.49, "negative"), (0.5, "positive"), (0.9, "positive")]
)
def test_real_provider_compiles_choice_and_noul_in_one_call(probability, expected):
    client = provider("negative")
    client.system_one.return_value.answers["positive_probability"] = NoulAnswer(
        noul=probability
    )
    row = Sentence("yelp:1", "yelp", "Not good.", "negative")
    record = asyncio.run(predict(mocked_agent(client, compare_noul=True), row))
    client.system_one.assert_awaited_once()
    state, questions = client.system_one.call_args.args
    assert set(questions) == {"sentiment", "positive_probability"}
    assert questions["sentiment"].type == "choice"
    assert questions["positive_probability"].type == "noul"
    assert "gold" not in str(state) and "yelp" not in str(state)
    assert record["choice"] == "negative"
    assert record["noul"]["positive_probability"] == probability
    assert record["noul"]["choice"] == expected
    assert "confidence" not in record["noul"]
    assert record["usage"]["requests"] == 1


def test_paired_comparison_metrics():
    records = [
        {
            "id": str(i),
            "source": "imdb",
            "gold": "positive",
            "choice": choice,
            "noul": {"choice": noul},
        }
        for i, (choice, noul) in enumerate(
            [
                ("positive", "positive"),
                ("positive", "negative"),
                ("negative", "positive"),
                ("negative", "negative"),
            ]
        )
    ]
    result = comparison_metrics(records)
    assert result["agreement_rate"] == 0.5
    assert result["choice"]["accuracy"] == result["noul"]["accuracy"] == 0.5
    assert result["noul_minus_choice_accuracy"] == 0
    assert result["paired_correctness"] == {
        "both_correct": 1,
        "choice_only_correct": 1,
        "noul_only_correct": 1,
        "both_wrong": 1,
    }
    assert result["disagreement_ids"] == ["1", "2"]
    assert result["per_source_noul"]["imdb"]["accuracy"] == 0.5
    assert comparison_metrics([])["agreement_rate"] is None


def test_comparison_report_is_persisted(tmp_path):
    client = provider("negative")
    client.system_one.return_value.answers["positive_probability"] = NoulAnswer(
        noul=0.9
    )
    summary = asyncio.run(
        run(
            sample_rows()[:2],
            output=tmp_path / "paired.jsonl",
            manifest={"compare_noul": True},
            agent=mocked_agent(client, compare_noul=True),
        )
    )
    assert summary["comparison"]["choice"]["accuracy"] == 1
    assert summary["comparison"]["noul"]["accuracy"] == 0
    assert summary["comparison"]["noul_minus_choice_accuracy"] == -1
    assert summary["agent_runs_attempted"] == client.system_one.await_count == 2


def test_metrics():
    records = [
        {"gold": gold, "choice": choice}
        for gold, choice in [
            ("positive", "positive"),
            ("negative", "negative"),
            ("negative", "positive"),
        ]
    ]
    result = metrics(records)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"]["negative"]["positive"] == 1
    assert metrics([])["accuracy"] is None


def test_inference_outputs_and_exclusive_paths(tmp_path, capsys):
    output = tmp_path / "predictions.jsonl"
    rows = select_sentences(sample_rows(), source="all", limit=6, seed=42)
    client = provider("negative")
    agent = mocked_agent(client)
    summary = asyncio.run(
        run(rows, output=output, manifest={"dataset_sha256": SHA256}, agent=agent)
    )
    assert summary["status"] == "complete"
    assert summary["agent_runs_attempted"] == client.system_one.await_count == 6
    assert summary["metrics"]["count"] == 6
    assert len(output.read_text().splitlines()) == 6
    assert "choice" in json.loads(output.read_text().splitlines()[0])
    progress = capsys.readouterr().err
    assert "Validation inference" in progress
    assert "6/6" in progress
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(run(rows, output=output, manifest={}, agent=agent))


def test_partial_run_flushes_progress_and_marks_failure(tmp_path):
    output = tmp_path / "partial.jsonl"
    rows = sample_rows()[:2]
    client = provider("negative")
    response = client.system_one.return_value
    client.system_one.side_effect = [response, RuntimeError("provider unavailable")]
    with pytest.raises(RuntimeError):
        asyncio.run(run(rows, output=output, manifest={}, agent=mocked_agent(client)))
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["status"] == "partial"
    assert summary["completed_count"] == 1
    assert summary["agent_runs_attempted"] == 2
    assert len(output.read_text().splitlines()) == 1


def mock_cli_provider(monkeypatch, *, compare_noul=False):
    client = provider("negative")
    if compare_noul:
        client.system_one.return_value.answers["positive_probability"] = NoulAnswer(
            noul=0.1
        )
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test-key")
    monkeypatch.setattr(
        "pydantic_ai.models.infer_model",
        lambda model: infer_model(
            model,
            provider_factory=lambda provider: TypeSafeProvider(typesafe_client=client),
        ),
    )
    return client


def test_cli_always_runs_inference_on_validation(monkeypatch, tmp_path):
    client = mock_cli_provider(monkeypatch)
    dataset = validation_fixture() + sample_rows()
    monkeypatch.setattr(
        "sentiment_benchmark.__main__.load_dataset", lambda path: dataset
    )
    assert main(["--limit", "6", "--output", str(tmp_path / "predictions.jsonl")]) == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert all(
        split_for(
            Sentence(**{key: row[key] for key in ("id", "source", "text", "gold")})
        )
        == "validation"
        for row in records
    )
    manifest = json.loads((tmp_path / "predictions.summary.json").read_text())[
        "manifest"
    ]
    assert client.system_one.await_count == 6
    assert manifest["split"] == "validation"
    assert manifest["validation_pool_count"] < manifest["dataset_count"]
    assert {row["id"] for row in records} <= set(manifest["validation_ids"])


@pytest.mark.parametrize("source", ["all", "imdb"])
def test_all_validation_includes_every_eligible_row(monkeypatch, tmp_path, source):
    client = mock_cli_provider(monkeypatch, compare_noul=True)
    dataset = validation_fixture() + sample_rows()
    monkeypatch.setattr(
        "sentiment_benchmark.__main__.load_dataset", lambda path: dataset
    )
    output = tmp_path / "all.jsonl"
    assert (
        main(
            [
                "--max-calls",
                "1000",
                "--compare-noul",
                "--all-validation",
                "--source",
                source,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    expected = {
        row.id
        for row in validation_sentences(dataset)
        if source == "all" or row.source == source
    }
    assert {row["id"] for row in records} == expected
    assert len(records) == len(expected)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["manifest"]["all_validation"]
    assert summary["manifest"]["selection_version"] == "all-validation-v1"
    assert summary["manifest"]["seed"] is None
    assert (
        summary["agent_runs_attempted"]
        == client.system_one.await_count
        == len(expected)
    )


def test_all_validation_checks_budget_before_constructing_client(monkeypatch, tmp_path):
    def forbidden(**kwargs):
        pytest.fail("Over-budget plan constructed a provider")

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test-key")
    monkeypatch.setattr(
        "sentiment_benchmark.__main__.load_dataset", lambda path: validation_fixture()
    )
    monkeypatch.setattr("sentiment_benchmark.__main__.make_agent", forbidden)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--all-validation",
                "--max-calls",
                "1",
                "--output",
                str(tmp_path / "unused.jsonl"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "unused.jsonl").exists()


def test_all_validation_and_limit_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--all-validation",
                "--limit",
                "6",
                "--output",
                str(tmp_path / "unused.jsonl"),
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--allow-paid", "--dry-run"])
def test_removed_flags_are_rejected(flag, tmp_path):
    with pytest.raises(SystemExit) as error:
        main([flag, "--output", str(tmp_path / "unused.jsonl")])
    assert error.value.code == 2


def test_cli_requires_key_and_budget(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    for flags in ([], ["--limit", "60"]):
        with pytest.raises(SystemExit) as error:
            main([*flags, "--output", str(tmp_path / "unused.jsonl")])
        assert error.value.code == 2
    assert not (tmp_path / "unused.jsonl").exists()
