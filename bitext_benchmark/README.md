# Jev on Bitext customer support

Single-message, zero-shot intent classification using **only Jev through
PydanticAI**. No fallback LLM, training, response generation, or transformers.
Data loading and scoring use the Python standard library.

## Run from the repository root

```bash
uv sync
# Public data download + request preparation; no API key or inference:
uv run python -m bitext_benchmark --download --dry-run \
  --output results/bitext/dev-inputs.jsonl

# After estimating cost and approving a budget:
export TYPESAFE_API_KEY=...
uv run python -m bitext_benchmark --limit 20 --allow-paid \
  --output results/bitext/dev-predictions.jsonl

# Fixed test evaluation (potentially thousands of paid calls):
uv run python -m bitext_benchmark --split test --full-split --allow-paid \
  --output results/bitext/test-predictions.jsonl

uv run pytest bitext_benchmark/tests -q
```

Default: 20 hash-ordered development examples, seed 42, sequential inference,
`typesafe:jev-1.13.0`. `--model` accepts only `typesafe:jev-*`. Use a fresh output
path for every run. `--data path/to/file.csv` permits a local CSV without download;
its actual hash and path are recorded, not falsely attributed to the pinned URL.

### Programmatic configuration

```python
import asyncio
from pathlib import Path
from bitext_benchmark.config import BenchmarkConfig
from bitext_benchmark.evaluation import run_evaluation

config = BenchmarkConfig(
    output=Path('results/bitext/programmatic-inputs.jsonl'),
    download=True,
    dry_run=True,
    limit=20,
)
asyncio.run(run_evaluation(config))
```

`BenchmarkConfig` owns defaults, positive-limit and split validation, Jev-only
model selection, explicit paid authorization, `limit`/`full_split` exclusivity,
download-path constraints, and artifact-path preflight. It is immutable and
rejects unknown fields. Omit `limit` when setting `full_split=True`. The CLI
retains the same flags; argparse only collects values and displays help.
Dataset-content checks remain in the loader, and the runner rechecks artifact
paths immediately before exclusive file creation.

## Dataset and provenance

- [Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Reference revision: `430d1a89bd93bd1fa23c16f29dd53e73f0087443`
- Source: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- License: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)
- Attribution: Bitext Innovations, 2024.

The pinned CSV contains **26,872 rows, 27 intents, and 11 category values**.
The upstream card says 10 categories and its intent list is incomplete; the
implementation's complete vocabulary was verified against the CSV. Inputs are
hybrid synthetic examples, not real customer conversations.

| Field | Role |
| --- | --- |
| `instruction` | Only per-example model input |
| `intent` | Local ground truth |
| `category` | Retained evaluation metadata, never model input |
| `flags` | Language-variation slice metadata, never model input |
| `response` | Not used or retained in prediction artifacts |

Pinned downloads are cached under `data/bitext/<revision>/`. Review the license
before redistributing source or derived data; retain required notices.

## Evaluation protocol

This is a custom evaluation of a training CSV, **not an official held-out split**.
No examples are used to train a model. Normalize messages with Unicode NFKC,
case folding, and whitespace collapsing. SHA-256 of seed + normalized message
assigns approximately 20% to development and 80% to test. Exact normalized
matches stay together; conflicting labels fail validation. The split is not
stratified. Synthetic near-duplicates/templates can still cross splits, so
results do not establish generalization to real support traffic.

Keep seed and source fixed; tune prompts on development only. Summaries save
source hash, split algorithm version, full split row IDs, selected IDs, label
counts, and duplicate counts. Row IDs refer to zero-based source CSV records.
Unknown labels, missing columns, and empty instructions fail validation.

The typed `Prediction.intent` enum becomes a single Jev Choice with 27 described
options. Per-example state contains only `instruction`. No no-match label is
added because this is a closed-set benchmark.

Reports include accuracy, all-27-label macro-F1, supported-label macro-F1,
per-intent precision/recall/F1, confusion counts, and per-language-tag accuracy.
Absent labels contribute zero to all-label macro-F1; small-sample macro-F1 should
not be compared with full-split results. Confidence is diagnostic, not a validated
routing threshold.

## Artifacts and safeguards

JSONL predictions include gold/predicted labels, confidence, probabilities,
provider details, resolved model, token usage, and latency. Dry-run JSONL contains
prepared state and local evaluation metadata, not predictions. A companion
`.summary.json` records provenance and metrics. Existing files are never
intentionally overwritten. Successful rows flush immediately; Python exceptions
produce partial summaries and are re-raised. Forced termination may leave only
JSONL. No resume support is provided.

`--allow-paid` is explicit authorization, **not a monetary cap**. The row count
bounds agent runs; SDK retries can add network requests and charges. Estimate
cost using dry-run input sizes, the 27-option schema, and current TypeSafe pricing
before authorizing a run. Dry runs never instantiate an agent or make inference
calls, but `--download` can fetch public data. Tests mock model responses and make
no paid calls.

## Layout

- `config.py`: shared constants, defaults, and immutable Pydantic `BenchmarkConfig`
- `cli.py`: CLI flags derived from model fields; raw arguments validated by the model
- `evaluation.py`: config-driven dataset preparation and orchestration
- `data.py`: pinned download, validation, grouped splitting and sampling
- `schema.py`: typed intent vocabulary and question descriptions
- `model.py`: PydanticAI/Jev inference and response normalization
- `metrics.py`: scoring and language-tag slices
- `runner.py`: flushed JSONL and partial summaries
- `__main__.py`: minimal module entry point
- `tests/`: offline regression tests, including real-provider request compilation
