"""Thin argparse adapter; defaults and validation belong to BenchmarkConfig."""
import argparse
import asyncio
import json

from pydantic import ValidationError

from .config import BenchmarkConfig, CLI_DESCRIPTION
from .evaluation import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Derive flags and help from model fields, leaving raw values to Pydantic."""
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    for name, field in BenchmarkConfig.model_fields.items():
        help_text = field.description or ''
        if not field.is_required():
            help_text += f' (default: {field.default})'
        options = {'dest': name, 'default': argparse.SUPPRESS, 'help': help_text}
        if field.annotation is bool:
            options['action'] = 'store_true'
        parser.add_argument(f'--{name.replace("_", "-")}', **options)
    return parser


def parse_config(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = build_parser()
    try:
        # Suppressed argparse defaults preserve explicit --limit detection.
        return BenchmarkConfig.model_validate(vars(parser.parse_args(argv)))
    except ValidationError as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> None:
    config = parse_config(argv)
    try:
        summary = asyncio.run(run_evaluation(config))
    except (ValueError, OSError) as error:
        build_parser().error(str(error))
    print(json.dumps({'status': summary['status'], 'metrics': summary['metrics']}, indent=2))
