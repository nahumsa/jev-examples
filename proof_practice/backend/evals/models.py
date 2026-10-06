from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.grading import RUBRIC
from app.models import Evaluation, Submission


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    submission: Submission
    second_proof: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
        | None
    ) = None


class EvalOutput(BaseModel):
    answer_a: Evaluation
    answer_b: Evaluation | None = None


class ScoreBand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum: float = Field(default=0, ge=0, le=2)
    maximum: float = Field(default=2, ge=0, le=2)

    @model_validator(mode="after")
    def ordered(self) -> "ScoreBand":
        if self.minimum > self.maximum:
            raise ValueError("Score band minimum exceeds maximum")
        return self


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    rationale: str
    answer_a: dict[str, ScoreBand]
    answer_b: dict[str, ScoreBand] = Field(default_factory=dict)
    minimum_total_improvement: float | None = Field(default=None, ge=0, le=10)
    minimum_logic_improvement: float | None = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def known_dimensions(self) -> "Expectations":
        if not self.answer_a:
            raise ValueError("Each case needs at least one labeled dimension")
        if (set(self.answer_a) | set(self.answer_b)) - set(RUBRIC):
            raise ValueError("Unknown rubric dimension")
        return self
