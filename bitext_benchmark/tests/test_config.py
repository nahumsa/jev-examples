"""Configuration validation and CLI adaptation without network calls."""
import asyncio
import csv
import json

import pytest
from pydantic import ValidationError

from bitext_benchmark.cli import main, parse_config
from bitext_benchmark.config import (
    BenchmarkConfig, CSV_COLUMNS, DEFAULT_LIMIT, DEFAULT_MODEL, DEFAULT_PATH,
    DEFAULT_SEED, DEFAULT_SPLIT,
)
from bitext_benchmark.evaluation import run_evaluation


def test_config_defaults(tmp_path):
    config = BenchmarkConfig(output=tmp_path / 'inputs.jsonl', dry_run=True)
    assert config.data == DEFAULT_PATH
    assert config.model == DEFAULT_MODEL
    assert config.seed == DEFAULT_SEED
    assert config.split == DEFAULT_SPLIT
    assert config.effective_limit == DEFAULT_LIMIT
    assert config.summary_path == tmp_path / 'inputs.summary.json'
    with pytest.raises(ValidationError):
        config.seed = 7


@pytest.mark.parametrize('options,match', [
    ({'limit': 0}, 'greater than 0'),
    ({'limit': -1}, 'greater than 0'),
    ({'limit': 'not-a-number'}, 'valid integer'),
    ({'split': 'train'}, 'dev'),
    ({'seed': 'not-a-number'}, 'valid integer'),
    ({'model': 'openai:gpt-4o'}, 'Only typesafe'),
    ({'model': 'typesafe:jev-'}, 'Only typesafe'),
    ({'full_split': True, 'limit': 20}, 'mutually exclusive'),
    ({'full_split': True, 'limit': 10}, 'mutually exclusive'),
    ({'extra_option': True}, 'Extra inputs'),
])
def test_invalid_options(tmp_path, options, match):
    with pytest.raises(ValidationError, match=match):
        BenchmarkConfig(output=tmp_path / 'inputs.jsonl', dry_run=True, **options)


def test_paid_authorization(tmp_path):
    with pytest.raises(ValidationError, match='requires --allow-paid'):
        BenchmarkConfig(output=tmp_path / 'predictions.jsonl')
    assert BenchmarkConfig(output=tmp_path / 'predictions.jsonl', allow_paid=True).allow_paid


def test_full_split_uses_model_default_without_conflict(tmp_path):
    config = BenchmarkConfig(output=tmp_path / 'inputs.jsonl', dry_run=True, full_split=True)
    assert config.effective_limit is None


def test_required_output():
    with pytest.raises(ValidationError, match='output'):
        BenchmarkConfig(dry_run=True)


@pytest.mark.parametrize('existing_suffix', ['.jsonl', '.summary.json'])
def test_existing_artifacts(tmp_path, existing_suffix):
    (tmp_path / f'inputs{existing_suffix}').write_text('keep me')
    with pytest.raises(ValidationError, match='already exists'):
        BenchmarkConfig(output=tmp_path / 'inputs.jsonl', dry_run=True)
    assert (tmp_path / f'inputs{existing_suffix}').read_text() == 'keep me'


def test_output_summary_collision(tmp_path):
    with pytest.raises(ValidationError, match='suffix'):
        BenchmarkConfig(output=tmp_path / 'inputs.summary.json', dry_run=True)


def test_download_requires_pinned_path_only_when_missing(tmp_path):
    custom = tmp_path / 'custom.csv'
    with pytest.raises(ValidationError, match='default pinned cache'):
        BenchmarkConfig(data=custom, download=True, output=tmp_path / 'inputs.jsonl', dry_run=True)
    custom.write_text('existing file')
    assert BenchmarkConfig(data=custom, download=True, output=tmp_path / 'inputs.jsonl', dry_run=True)


def test_parse_config_converts_raw_values(tmp_path):
    config = parse_config(['--output', str(tmp_path / 'inputs.jsonl'), '--dry-run',
                           '--limit', '5000', '--seed', '7', '--split', 'test'])
    assert config.limit == 5000
    assert config.seed == 7
    assert config.output == tmp_path / 'inputs.jsonl'
    assert config.split == 'test'
    full = parse_config(['--output', str(tmp_path / 'full.jsonl'), '--dry-run', '--full-split'])
    assert full.effective_limit is None


@pytest.mark.parametrize('flags', [
    ['--limit', '0'], ['--split', 'train'], ['--model', 'openai:gpt-4o'],
    ['--full-split', '--limit', '20'],
])
def test_cli_validation_uses_model(tmp_path, flags, capsys):
    with pytest.raises(SystemExit) as error:
        parse_config(['--output', str(tmp_path / 'inputs.jsonl'), '--dry-run', *flags])
    assert error.value.code == 2
    assert 'BenchmarkConfig' in capsys.readouterr().err
    assert not (tmp_path / 'inputs.jsonl').exists()


def test_cli_does_not_load_data_for_invalid_config(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid configuration reached dataset loading')
    monkeypatch.setattr('bitext_benchmark.evaluation.load_data', forbidden)
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path / 'inputs.jsonl'), '--limit', '0', '--dry-run'])


def test_config_driven_dry_run(tmp_path):
    path = tmp_path / 'data.csv'
    with path.open('w', newline='') as source:
        writer = csv.DictWriter(source, fieldnames=sorted(CSV_COLUMNS))
        writer.writeheader()
        for index in range(100):
            writer.writerow({'instruction': f'Cancel my order {index}', 'intent': 'cancel_order',
                             'category': 'ORDER', 'flags': 'B', 'response': 'EXCLUDED'})
    config = BenchmarkConfig(data=path, output=tmp_path / 'inputs.jsonl', dry_run=True, limit=5)
    summary = asyncio.run(run_evaluation(config))
    assert summary['completed_count'] == 5
    assert summary['status'] == 'complete'
    assert summary['manifest']['seed'] == DEFAULT_SEED
    assert summary['metrics'] is None
    assert json.loads(config.summary_path.read_text()) == summary
    assert 'EXCLUDED' not in config.output.read_text()
