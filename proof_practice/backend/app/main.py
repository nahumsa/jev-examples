import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic_settings import BaseSettings, SettingsConfigDict
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from .grading import RUBRIC, GradingDeps, JevClient, proof_agent
from .models import Comparison, ComparisonSubmission, Evaluation, Problem, Submission
from .problems import BY_ID, PROBLEMS

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    typesafe_api_key: str | None = None
    typesafe_default_model: str = "jev-latest"
    typesafe_base_url: str = "https://api.typesafe.ai"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.jev = None
    if settings.typesafe_api_key:
        app.state.jev = AsyncTypeSafeClient(
            api_key=settings.typesafe_api_key,
            model=settings.typesafe_default_model,
            base_url=settings.typesafe_base_url,
            timeout=25,
            retry=RetryPolicy(max_retries=1),
        )
    try:
        yield
    finally:
        if app.state.jev is not None:
            await app.state.jev.aclose()


app = FastAPI(title="Proof Practice", lifespan=lifespan)


def get_jev(request: Request) -> JevClient:
    client = request.app.state.jev
    if client is None:
        raise HTTPException(503, "Grading is not configured. Set TYPESAFE_API_KEY on the backend.")
    return client


@app.post("/api/compare", response_model=Comparison)
async def compare(
    submission: ComparisonSubmission, client: Annotated[JevClient, Depends(get_jev)]
) -> Comparison:
    answer_a = Submission.model_validate(submission.model_dump(exclude={"second_proof"}))
    answer_b = answer_a.model_copy(update={"proof": submission.second_proof})
    # Independent agent runs prevent either answer from influencing the other.
    tasks = [
        asyncio.create_task(evaluate_submission(answer_a, client)),
        asyncio.create_task(evaluate_submission(answer_b, client)),
    ]
    try:
        a, b = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return Comparison(answer_a=a, answer_b=b, total_difference=b.total - a.total)


@app.get("/api/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "grading_configured": request.app.state.jev is not None}


@app.get("/api/problems", response_model=list[Problem])
async def problems() -> list[Problem]:
    return PROBLEMS


@app.get("/api/rubric")
async def rubric() -> dict:
    return RUBRIC


@app.post("/api/evaluate", response_model=Evaluation)
async def evaluate(
    submission: Submission, client: Annotated[JevClient, Depends(get_jev)]
) -> Evaluation:
    return await evaluate_submission(submission, client)


async def evaluate_submission(submission: Submission, client: JevClient) -> Evaluation:
    if submission.custom_statement is not None:
        problem = Problem(
            id="custom",
            title="Custom question",
            topic="Custom",
            statement=submission.custom_statement,
            hint="",
        )
    else:
        problem = BY_ID.get(submission.problem_id)
        if problem is None:
            raise HTTPException(404, "Unknown problem")
    try:
        async with asyncio.timeout(55):
            result = await proof_agent.run(
                "Grade the proof provided in dependencies.",
                deps=GradingDeps(client, problem, submission.proof),
            )
        return result.output
    except TimeoutError:
        raise HTTPException(504, "Grading timed out. Please try again.") from None
    except (TypeSafeError, ValueError):
        # Do not log submitted proof text, credentials, or provider response bodies.
        logger.warning("Jev grading failed")
        raise HTTPException(
            502, "The grading service could not evaluate this proof. Try again."
        ) from None
