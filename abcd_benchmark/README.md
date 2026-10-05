# Jev on ABCD

ABCD code is grouped by feature instead of scattered across the package root:

```text
abcd_benchmark/
  ast_cds/    AST/CDS examples, requests, inference, scoring, and orchestration
  intent/     Conversation-prefix intent routing
  common/     Shared dataset I/O, sampling, and artifact utilities
  cli/        Command-line entry points
  docs/       Protocol documentation and coding guidelines
  tests/
    ast_cds/  Benchmark tests and pinned upstream fixtures
    intent/   Intent-routing tests
  __main__.py AST/CDS command entry point
```

Run commands **from the repository root**. Cached datasets stay in `data/abcd/`,
outputs in `results/`, and project tooling remains in the repository root.

## Setup

```bash
uv sync
uv run python -m abcd_benchmark.cli.download
# Optional uncompressed dataset and response-candidate lookup table:
uv run python -m abcd_benchmark.cli.download --unpack --include-utterances
```

Downloads use pinned ABCD revision `6b8700ce67c6b37b062dd7a60abc76d7ef832a97`.
Existing files are reused; the manifest records hashes and split counts. The full
dataset has 8,034 train, 1,004 dev, and 1,004 test conversations.

## Prepare requests without inference

```bash
uv run python -m abcd_benchmark.cli.intent --split sample --dry-run \
  --output results/sample-inputs.jsonl
uv run python -m abcd_benchmark --task ast --split dev --limit 1 --dry-run \
  --output results/ast-inputs.jsonl
uv run python -m abcd_benchmark --task cds --split dev --limit 1 \
  --preprocessing raw --dry-run --output results/cds-raw-inputs.jsonl
```

`python -m abcd_benchmark.cli.benchmark` is equivalent to
`python -m abcd_benchmark`. Dry runs make no model calls but may download missing
public data or the reference BERT vocabulary. Raw mode never loads that tokenizer.

## Paid inference

Set `TYPESAFE_API_KEY`, estimate cost, and get budget approval **before** removing
`--dry-run`. There is no hard monetary cap or resume support in the current runner.
Intent routing accepts names such as `typesafe:jev-1.13.0`; AST/CDS uses
`jev-1.13.0` without the prefix. No training or fine-tuning is performed.

Use small dev samples for exploration and reserve test for fixed evaluations.
`--full-split` selects an entire split. Intent routing defaults to 20 seeded dev
conversations, three customer messages, and one parallel call. Zero customer-turn
limit includes all speech and is retrospective. AST/CDS defaults to three dev
conversations and four concurrent network requests.

## Offline CDS subset analysis

```bash
uv run python -m abcd_benchmark.cli.summarize_cds \
  --inputs results/cds-test-raw.jsonl results/cds-test-full.jsonl \
  --output results/cds-subset-analysis.json
```

Only conversations with exact expected example-ID coverage are scored. Reports
include individual completed subsets and matched-conversation comparisons. This
reads saved predictions locally and makes no model calls.

## Imports and architecture

Use `abcd_benchmark.ast_cds.*` for benchmark modules, `abcd_benchmark.intent.*` for
intent routing, and `abcd_benchmark.common.*` for shared utilities. The former flat
benchmark modules have moved into `ast_cds/`; `jev.py` is now `pipeline.py`.
Intent adapters are `intent.evaluation`, `intent.model`, and `intent.summary`.
Shared `data.py` is now `common.data`. Former flat CLI module names have moved to
`cli.benchmark`, `cli.intent`, `cli.download`, and `cli.summarize_cds`.

Function interfaces, CLI flags, prompts, label ordering, output fields, and scoring
rules remain unchanged. See [protocol details](docs/BENCHMARKS.md) and
[modular-code guidelines](docs/MODULAR_CODE_GUIDELINES.md).

## Artifacts and safeguards

JSONL retains predictions, confidence, usage, latency, and provenance. Successful
rows are flushed; normal Python failures produce partial summaries. Forced
termination may leave only JSONL. Existing outputs must not be overwritten.

Only permitted observed inputs go to inference; labels and scenario metadata stay
outside model state. Confidence is diagnostic, not a validated routing threshold.
Estimated costs use $0.042 per million input tokens with free output and exclude
retries and unsaved calls; they are not invoices.

## Tests

```bash
uv run pytest -q
uv run pytest abcd_benchmark/tests/ast_cds -q
uv run pytest abcd_benchmark/tests/intent -q
```

Tests make no paid API calls. Data/results are ignored by Git. Retain ABCD's
upstream provenance and citation when using its data.
