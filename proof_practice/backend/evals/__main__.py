import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

from pydantic import TypeAdapter
from pydantic_evals.reporting import EvaluationReport
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from app.grading import RUBRIC
from app.main import Settings

from .dataset import CASES_PATH, ProofDataset, load_dataset
from .models import EvalInput, EvalOutput, Expectations
from .task import make_task

Report = EvaluationReport[EvalInput, EvalOutput, Expectations]
REPORT_ADAPTER = TypeAdapter(Report)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate Proof Practice with Pydantic Evals")
    result.add_argument(
        "--list", action="store_true", help="List cases and call count; no inference"
    )
    result.add_argument("--allow-paid", action="store_true", help="Authorize live Jev API calls")
    result.add_argument(
        "--case", action="append", default=[], help="Select a named case; repeatable"
    )
    result.add_argument("--limit", type=positive, help="Take the first N selected cases")
    result.add_argument("--repeat", type=positive, default=1)
    result.add_argument(
        "--max-calls", type=positive, help="Refuse plans above this SDK-call budget"
    )
    result.add_argument("--concurrency", type=positive, default=1)
    result.add_argument(
        "--model", help="Override TYPESAFE_DEFAULT_MODEL; pin for reproducible runs"
    )
    result.add_argument("--output", type=Path, help="Required for live runs; exclusive JSON report")
    return result


def report_passed(report: Report) -> bool:
    return (
        not report.failures
        and not report.report_evaluator_failures
        and bool(report.cases)
        and all(
            not case.evaluator_failures
            and bool(case.assertions)
            and all(assertion.value for assertion in case.assertions.values())
            for case in report.cases
        )
    )


async def run_live(
    dataset: ProofDataset, settings: Settings, args: argparse.Namespace, output: TextIO
) -> bool:
    model = args.model or settings.typesafe_default_model
    async with AsyncTypeSafeClient(
        api_key=settings.typesafe_api_key,
        base_url=settings.typesafe_base_url,
        model=model,
        timeout=25,
        retry=RetryPolicy(max_retries=0),
    ) as client:
        report = await dataset.evaluate(
            make_task(client),
            name="proof-practice-quality",
            max_concurrency=args.concurrency,
            repeat=args.repeat,
            progress=False,
            metadata={
                "requested_model": model,
                "dataset_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
                "rubric_sha256": hashlib.sha256(
                    json.dumps(RUBRIC, sort_keys=True).encode()
                ).hexdigest(),
                "pydantic_evals_version": version("pydantic-evals"),
                "typesafe_sdk_version": version("typesafe-sdk"),
                "pydantic_ai_version": version("pydantic-ai-slim"),
                "source_sha256": {
                    str(path.relative_to(Path(__file__).parent.parent)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for folder in ("app", "evals")
                    for path in sorted((Path(__file__).parent.parent / folder).glob("*.py"))
                },
                "repeat": args.repeat,
                "concurrency": args.concurrency,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "labels": "hand-authored starter expectations; not expert-validated ground truth",
            },
        )
    json.dump(REPORT_ADAPTER.dump_python(report, mode="json"), output, indent=2)
    output.write("\n")
    output.flush()
    report.print(include_reasons=True, include_error_stacktrace=False)
    return report_passed(report)


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    dataset = load_dataset()
    known = {case.name for case in dataset.cases}
    if set(args.case) - known:
        cli.error("Unknown case name; use --list to see available cases")
    if args.case:
        dataset.cases[:] = [case for case in dataset.cases if case.name in args.case]
    if args.limit:
        dataset.cases[:] = dataset.cases[: args.limit]
    calls = sum(2 if case.inputs.second_proof is not None else 1 for case in dataset.cases)
    planned_calls = calls * args.repeat
    print(f"{len(dataset.cases)} cases × {args.repeat} repeats; {planned_calls} Jev calls.")
    if args.max_calls is not None and planned_calls > args.max_calls:
        cli.error(
            f"Plan requires {planned_calls} Jev calls, exceeding --max-calls {args.max_calls}"
        )
    if args.list:
        for case in dataset.cases:
            print(f"  {case.name} ({case.metadata.category})")
        print("No inference performed. Labels and thresholds are provisional study expectations.")
        return 0
    if not args.allow_paid:
        cli.error("Live evaluation requires --allow-paid. Use --list for a no-cost preview.")
    if args.output is None:
        cli.error("Live evaluation requires --output to retain results")
    if args.output.exists():
        cli.error("Output already exists; choose a new path")
    settings = Settings()
    if not settings.typesafe_api_key:
        cli.error("Set TYPESAFE_API_KEY in the backend environment or .env")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as output:
            passed = asyncio.run(run_live(dataset, settings, args, output))
    except FileExistsError:
        cli.error("Output already exists; choose a new path")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
