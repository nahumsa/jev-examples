"""Shared constants and validated benchmark options; no writes or inference."""
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATASET = 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
REVISION = '430d1a89bd93bd1fa23c16f29dd53e73f0087443'
FILENAME = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}'
DEFAULT_PATH = Path('data/bitext') / REVISION / FILENAME
DEFAULT_MODEL = 'typesafe:jev-1.13.0'
DEFAULT_SPLIT = 'dev'
DEFAULT_SEED = 42
DEFAULT_LIMIT = 20
REQUEST_TIMEOUT_SECONDS = 60
AGENT_RETRIES = 0
SUMMARY_SUFFIX = '.summary.json'
CSV_COLUMNS = frozenset({'instruction', 'intent', 'category', 'flags', 'response'})
SPLIT_BUCKETS = 100
DEV_BUCKETS = 20
SPLIT_VERSION = 'normalized-sha256-20dev-80test-v1'
Split = Literal['dev', 'test']
SPLITS: tuple[Split, ...] = ('dev', 'test')
CLI_DESCRIPTION = 'Bitext intent benchmark: Jev via PydanticAI only'


def validate_model(model: str) -> str:
    """Shared guard for config validation and direct agent construction."""
    if not model.startswith('typesafe:jev-') or model == 'typesafe:jev-':
        raise ValueError('Only typesafe:jev-* models are supported')
    return model


def validate_output(output: Path) -> Path:
    """Preflight artifact paths; exclusive creation remains the runtime guard."""
    if output.name.endswith(SUMMARY_SUFFIX):
        raise ValueError('Output must not have the .summary.json suffix')
    summary_path = output.with_suffix(SUMMARY_SUFFIX)
    if output.exists() or summary_path.exists():
        raise FileExistsError('Output or summary already exists; choose a new output path')
    return output


class BenchmarkConfig(BaseModel):
    """CLI and programmatic options with defaults and cross-field validation."""

    model_config = ConfigDict(frozen=True, extra='forbid', validate_default=True)

    data: Path = Field(default=DEFAULT_PATH, description='Local CSV path; defaults to the pinned cache')
    download: bool = Field(default=False, description='Download the missing pinned public CSV')
    split: Split = Field(default=DEFAULT_SPLIT, description='Evaluation split: dev or test')
    seed: int = Field(default=DEFAULT_SEED, description='Deterministic split and sampling seed')
    limit: int = Field(default=DEFAULT_LIMIT, gt=0, description='Maximum number of examples; must be positive')
    full_split: bool = Field(default=False, description='Select the entire split; cannot be combined with --limit')
    model: str = Field(default=DEFAULT_MODEL, description='Jev model name: typesafe:jev-* only')
    output: Path = Field(description='New JSONL output path; existing artifacts cannot be overwritten')
    dry_run: bool = Field(default=False, description='Save inputs without API calls')
    allow_paid: bool = Field(default=False, description='Explicitly authorize potentially billable calls')

    @field_validator('model')
    @classmethod
    def check_model(cls, value: str) -> str:
        return validate_model(value)

    @field_validator('output')
    @classmethod
    def check_output(cls, value: Path) -> Path:
        try:
            return validate_output(value)
        except FileExistsError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode='after')
    def check_options(self) -> Self:
        if self.full_split and 'limit' in self.model_fields_set:
            raise ValueError('--limit and --full-split are mutually exclusive')
        if not self.dry_run and not self.allow_paid:
            raise ValueError('Use --dry-run first; paid inference requires --allow-paid after budget approval')
        if self.download and self.data != DEFAULT_PATH and not self.data.exists():
            raise ValueError('Automatic downloads require the default pinned cache path')
        return self

    @property
    def effective_limit(self) -> int | None:
        return None if self.full_split else self.limit

    @property
    def summary_path(self) -> Path:
        return self.output.with_suffix(SUMMARY_SUFFIX)
