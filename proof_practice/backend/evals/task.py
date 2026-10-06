from collections.abc import Awaitable, Callable

from pydantic_evals import increment_eval_metric, set_eval_attribute
from typesafe_sdk import SystemOneResponse

from app.grading import JevClient
from app.main import compare, evaluate_submission
from app.models import ComparisonSubmission

from .models import EvalInput, EvalOutput


class MeteredClient:
    """Measure actual SDK invocations without exposing labels to the agent."""

    def __init__(self, client: JevClient):
        self.client = client

    async def system_one(self, state: dict, questions: dict) -> SystemOneResponse:
        increment_eval_metric("jev_calls", 1)
        response = await self.client.system_one(state, questions)
        for field in ("input_tokens", "output_tokens"):
            count = getattr(response.usage, field)
            if count is not None:
                increment_eval_metric(field, count)
        set_eval_attribute("resolved_model", response.model)
        return response


def make_task(client: JevClient) -> Callable[[EvalInput], Awaitable[EvalOutput]]:
    async def grade_proof(inputs: EvalInput) -> EvalOutput:
        metered = MeteredClient(client)
        if inputs.second_proof is None:
            answer = await evaluate_submission(inputs.submission, metered)
            return EvalOutput(answer_a=answer)
        submission = ComparisonSubmission(
            **inputs.submission.model_dump(), second_proof=inputs.second_proof
        )
        result = await compare(submission, metered)
        return EvalOutput(answer_a=result.answer_a, answer_b=result.answer_b)

    return grade_proof
