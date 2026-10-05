from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import ScoreAnswer, SystemOneResponse, TypeSafeError, Usage

from app.grading import QUESTIONS, RUBRIC
from app.main import Settings, app, get_jev


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("app.main.Settings", lambda: Settings(_env_file=None))
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def fake_jev(score=1.5):
    response = SystemOneResponse(
        model="jev-test",
        usage=Usage(input_tokens=100, output_tokens=20),
        answers={
            key: ScoreAnswer(
                score=score,
                confidence=0.4,
                probabilities={0: 0.0, 1: 2.0 - score, 2: score - 1.0},
                legend=dict(enumerate(value["criteria"])),
            )
            for key, value in RUBRIC.items()
        },
    )
    return AsyncMock(system_one=AsyncMock(return_value=response))


def test_catalog_and_rubric(client):
    response = client.get("/api/problems")
    assert response.status_code == 200
    assert len(response.json()) == 6
    assert "reference_proof" not in response.text
    assert client.get("/api/rubric").json() == RUBRIC
    assert all(len(q.criteria) == 3 for q in QUESTIONS.values())


def test_agent_tool_and_output_validation(client):
    jev = fake_jev()
    app.dependency_overrides[get_jev] = lambda: jev
    response = client.post(
        "/api/evaluate",
        json={"problem_id": "even-sum", "proof": "  Let a=2r, b=2s. Then a+b=2(r+s), even.  "},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total"] == 7.5
    assert len(result["dimensions"]) == 5
    assert result["dimensions"][0]["score"] == 1.5
    jev.system_one.assert_awaited_once()
    kwargs = jev.system_one.call_args.kwargs
    assert kwargs["questions"] == QUESTIONS
    assert kwargs["state"]["student_proof"].startswith("Let")
    assert "reference_proof" in kwargs["state"]


@pytest.mark.parametrize("proof", ["", "   ", "x" * 8001])
def test_invalid_proof(client, proof):
    app.dependency_overrides[get_jev] = lambda: fake_jev()
    assert (
        client.post("/api/evaluate", json={"problem_id": "even-sum", "proof": proof}).status_code
        == 422
    )


def test_unknown_problem(client):
    jev = fake_jev()
    app.dependency_overrides[get_jev] = lambda: jev
    assert (
        client.post("/api/evaluate", json={"problem_id": "unknown", "proof": "test"}).status_code
        == 404
    )
    jev.system_one.assert_not_awaited()


def test_missing_key(client):
    assert (
        client.post("/api/evaluate", json={"problem_id": "even-sum", "proof": "test"}).status_code
        == 503
    )


@pytest.mark.parametrize("error,status", [(TypeSafeError("secret"), 502), (TimeoutError(), 504)])
def test_provider_errors_are_safe(client, error, status):
    jev = fake_jev()
    jev.system_one.side_effect = error
    app.dependency_overrides[get_jev] = lambda: jev
    response = client.post("/api/evaluate", json={"problem_id": "even-sum", "proof": "test"})
    assert response.status_code == status
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "question",
    [
        {"problem_id": "even-sum"},
        {"custom_statement": "Prove that 2+2=4."},
    ],
)
def test_compare_answers(client, question):
    jev = fake_jev()
    first = fake_jev(1.2).system_one.return_value
    second = fake_jev(1.8).system_one.return_value
    jev.system_one.side_effect = [first, second]
    app.dependency_overrides[get_jev] = lambda: jev
    response = client.post(
        "/api/compare",
        json={
            **question,
            "proof": "Answer A",
            "second_proof": "Answer B",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["answer_a"]["total"] == pytest.approx(6)
    assert result["answer_b"]["total"] == pytest.approx(9)
    assert result["total_difference"] == pytest.approx(3)
    assert jev.system_one.await_count == 2
    states = [call.kwargs["state"] for call in jev.system_one.call_args_list]
    assert states[0]["problem"] == states[1]["problem"]
    assert [s["student_proof"] for s in states] == ["Answer A", "Answer B"]


@pytest.mark.parametrize("second", ["", "  ", "x" * 8001])
def test_compare_invalid_second_answer(client, second):
    jev = fake_jev()
    app.dependency_overrides[get_jev] = lambda: jev
    assert (
        client.post(
            "/api/compare",
            json={
                "problem_id": "even-sum",
                "proof": "A",
                "second_proof": second,
            },
        ).status_code
        == 422
    )
    jev.system_one.assert_not_awaited()


def test_compare_provider_failure(client):
    jev = fake_jev()
    jev.system_one.side_effect = TypeSafeError("secret")
    app.dependency_overrides[get_jev] = lambda: jev
    response = client.post(
        "/api/compare",
        json={
            "problem_id": "even-sum",
            "proof": "A",
            "second_proof": "B",
        },
    )
    assert response.status_code == 502
    assert "secret" not in response.text


def test_custom_question(client):
    jev = fake_jev()
    app.dependency_overrides[get_jev] = lambda: jev
    response = client.post(
        "/api/evaluate",
        json={
            "custom_statement": "  Prove that 3 divides n³-n for every integer n.  ",
            "proof": "The product (n-1)n(n+1) contains a multiple of three.",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["problem_id"] == "custom"
    state = jev.system_one.call_args.kwargs["state"]
    assert state["problem"].startswith("Prove")
    assert state["reference_proof"] is None


@pytest.mark.parametrize(
    "question",
    [
        {},
        {"custom_statement": "   "},
        {"custom_statement": "x" * 2001},
        {"custom_statement": "A claim", "problem_id": "even-sum"},
    ],
)
def test_invalid_question_source(client, question):
    app.dependency_overrides[get_jev] = lambda: fake_jev()
    assert client.post("/api/evaluate", json={**question, "proof": "test"}).status_code == 422


def test_incomplete_answers_rejected(client):
    jev = fake_jev()
    jev.system_one.return_value.answers.pop("justification")
    app.dependency_overrides[get_jev] = lambda: jev
    assert (
        client.post("/api/evaluate", json={"problem_id": "even-sum", "proof": "test"}).status_code
        == 502
    )
