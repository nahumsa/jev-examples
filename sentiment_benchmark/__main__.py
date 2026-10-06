"""Run Jev sentiment inference on the validation dataset (API charges apply)."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from zipfile import BadZipFile

from pydantic_ai.exceptions import AgentRunError, UserError

from .data import (
    DEFAULT_DATA,
    SHA256,
    SOURCES,
    URL,
    VALIDATION_SPLIT_SEED,
    VALIDATION_SPLIT_VERSION,
    download,
    load_dataset,
    select_sentences,
    validation_sentences,
)
from .evaluation import run
from .model import NOUL_THRESHOLD, ComparisonPrediction, Prediction, make_agent


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--download", action="store_true", help="Download the pinned UCI ZIP"
    )
    parser.add_argument("--source", choices=("all", *SOURCES), default="all")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit", type=positive, default=30, help="Total balanced sample size"
    )
    selection.add_argument(
        "--all-validation",
        action="store_true",
        help="Use every validation sentence for the selected source(s), without balancing",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-noul",
        action="store_true",
        help="Compare Choice with a positive-sentiment Noul in the same request",
    )
    parser.add_argument("--max-calls", type=positive, default=30)
    parser.add_argument(
        "--model", default="jev-1.13.0", help="Jev model ID without provider prefix"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh predictions .jsonl path"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"jev-[A-Za-z0-9._-]+", args.model):
        parser.error(
            "Use a Jev model ID such as jev-1.13.0, without a typesafe: prefix"
        )
    if args.output.suffix != ".jsonl":
        parser.error("--output must end in .jsonl")
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        parser.error("Output/report already exists; choose a fresh path")
    if not args.all_validation and args.limit > args.max_calls:
        parser.error("Requested --limit exceeds --max-calls")
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        parser.error(
            "Set TYPESAFE_API_KEY in your environment; .env is not automatically loaded"
        )
    try:
        if args.download:
            download(args.data)
        dataset = load_dataset(args.data)
        validation = validation_sentences(dataset)
        if args.all_validation:
            rows = [
                row
                for row in validation
                if args.source == "all" or row.source == args.source
            ]
        else:
            rows = select_sentences(
                validation,
                source=args.source,
                limit=args.limit,
                seed=args.seed,
            )
        if not rows:
            parser.error("No validation sentences available for the selected source(s)")
        if len(rows) > args.max_calls:
            parser.error(
                f"Plan requires {len(rows)} calls, exceeding --max-calls {args.max_calls}"
            )
        print(
            f"{len(rows)} validation sentences (pool: {len(validation)}); "
            f"maximum {len(rows)} agent requests (transport retries may add HTTP attempts)."
        )
        manifest = {
            "dataset_url": URL,
            "dataset_sha256": SHA256,
            "selection_version": "all-validation-v1"
            if args.all_validation
            else "balanced-source-label-v1",
            "all_validation": args.all_validation,
            "split": "validation",
            "split_version": VALIDATION_SPLIT_VERSION,
            "split_seed": VALIDATION_SPLIT_SEED,
            "dataset_count": len(dataset),
            "validation_pool_count": len(validation),
            "validation_ids": [row.id for row in validation],
            "validation_source_label_counts": dict(
                Counter(f"{row.source}/{row.gold}" for row in validation)
            ),
            "source": args.source,
            "seed": None if args.all_validation else args.seed,
            "model": args.model,
            "typesafe_sdk_version": version("typesafe-sdk"),
            "pydantic_ai_version": version("pydantic-ai-slim"),
            "output_schema": (
                ComparisonPrediction if args.compare_noul else Prediction
            ).model_json_schema(),
            "compare_noul": args.compare_noul,
            "noul_threshold": NOUL_THRESHOLD if args.compare_noul else None,
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "started_at": datetime.now(UTC).isoformat(),
            "max_calls": args.max_calls,
            "agent_retries": 0,
            "transport_retries": "provider-default",
        }

        async def execute() -> dict:
            async with make_agent(args.model, compare_noul=args.compare_noul) as agent:
                return await run(
                    rows,
                    output=args.output,
                    manifest=manifest,
                    agent=agent,
                )

        summary = asyncio.run(execute())
    except (
        OSError,
        ValueError,
        KeyError,
        BadZipFile,
        AgentRunError,
        UserError,
    ) as error:
        # SDK exceptions may contain raw response bodies; don't print them or keys.
        if isinstance(error, (ValueError, FileNotFoundError, FileExistsError)):
            print(f"Cannot run benchmark: {error}")
        else:
            print(
                f"Benchmark failed ({type(error).__name__}); inspect any partial report."
            )
        return 1
    print(json.dumps(summary["metrics"], indent=2))
    if summary["comparison"] is not None:
        print(json.dumps(summary["comparison"], indent=2))
    print(f"Saved {args.output} and {args.output.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
