from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class Problem(BaseModel):
    id: str
    title: str
    topic: str
    statement: str
    hint: str


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problem_id: str | None = None
    custom_statement: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
        | None
    ) = None
    proof: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]

    @model_validator(mode="after")
    def one_question_source(self) -> "Submission":
        if (self.problem_id is None) == (self.custom_statement is None):
            raise ValueError("Provide either problem_id or custom_statement, not both")
        return self


class ComparisonSubmission(Submission):
    second_proof: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
    ]


class Dimension(BaseModel):
    id: str
    title: str
    score: float = Field(ge=0, le=2, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    probabilities: dict[int, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]
    criteria: list[str]
    likely_level: int = Field(ge=0, le=2)
    rubric_feedback: str
    revision_tip: str

    @model_validator(mode="after")
    def valid_distribution(self) -> "Dimension":
        if set(self.probabilities) != {0, 1, 2}:
            raise ValueError("Expected three rubric levels")
        if abs(sum(self.probabilities.values()) - 1) > 0.02:
            raise ValueError("Probabilities must sum to one")
        expected = sum(k * v for k, v in self.probabilities.items())
        if abs(expected - self.score) > 0.03:
            raise ValueError("Score must match the probability-weighted levels")
        return self


class Evaluation(BaseModel):
    problem_id: str
    model: str
    total: float = Field(ge=0, le=10)
    dimensions: list[Dimension] = Field(min_length=5, max_length=5)
    notice: str = (
        "AI practice feedback, not formal proof verification. Confidence describes the "
        "model's distribution, not the probability that your proof is correct. "
        "Feedback and revision tips are rubric-based templates, not generated explanations."
    )


class Comparison(BaseModel):
    answer_a: Evaluation
    answer_b: Evaluation
    total_difference: float
