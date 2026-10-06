# UCI sentiment benchmark

A toy benchmark comparing Jev's positive/negative predictions with UCI labels,
using **PydanticAI's native TypeSafe integration** via `Agent("typesafe:jev-1.13.0", ...)`.
PydanticAI creates and manages the API client; benchmark inference code does not
construct a TypeSafe SDK client directly. The typed sentiment enum is
compiled into a Choice; optional paired mode adds a Noul. There is no fallback
LLM, generated feedback, fine-tuning, or additional judge.

## Dataset and validation split

[Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
has 3,000 sentences: 1,000 each from Amazon product reviews (`amazon_cells`), IMDb
movie reviews (`imdb`), and Yelp restaurant reviews (`yelp`). Each source has 500
positive (`1`) and 500 negative (`0`) labels. This is **not Amazon Polarity**.

UCI provides no official validation split. This repository defines an approximately
20% holdout using SHA-256 of NFKC-normalized, case-folded, whitespace-collapsed text,
with fixed split seed 42. Normalized duplicates stay together across sources and
labels. Gold labels are not used to assign splits. The pinned archive produces
**589 validation sentences** and 2,411 development sentences. Only validation
sentences are eligible for CLI inference; original text is sent unchanged.

By default, samples are balanced by source and label, with a maximum all-source
sample of **546**. `--all-validation` includes all 589 rows without balancing.
`--seed` changes sampling within validation, never the holdout assignment. Reports
record the split version, seed, validation IDs, and source/label counts.

The ZIP is cached in ignored `data/sentiment/`. Its SHA-256 is pinned in `data.py`;
corrupt or changed archives are rejected. Only three known ZIP members are read,
without extraction. Parsing preserves quotes, tabs inside sentences, trailing
spaces, and Unicode separators that are not record boundaries.

Attribution: Kotzias, D. (2015), *Sentiment Labelled Sentences*, UCI Machine Learning
Repository, [doi:10.24432/C57604](https://doi.org/10.24432/C57604). Associated paper:
Kotzias, Denil, de Freitas, and Smyth, *From Group to Individual Labels Using Deep
Features*, KDD 2015. Consult UCI for licensing and terms.

## Setup and offline tests

Run from the repository root, using the shared root Python environment:

```sh
uv sync --locked
uv run pytest sentiment_benchmark/tests -q
uv run python -m sentiment_benchmark --help
```

Tests use synthetic responses, including the real PydanticAI provider adapter.
They exercise the harness without paid calls; they do not establish Jev's accuracy.
These tests are included in the repository's offline GitHub CI.

Set `TYPESAFE_API_KEY` in your shell environment before inference. `.env` is not
automatically loaded. Never commit credentials.

## Run inference

```sh
# Download the archive if missing, then compare six validation sentences.
uv run python -m sentiment_benchmark --download --compare-noul \
  --limit 6 --max-calls 6 --output results/sentiment/validation-smoke.jsonl

# Balanced sample: 300 validation sentences, Choice plus Noul.
uv run python -m sentiment_benchmark --compare-noul --limit 300 --max-calls 300 \
  --output results/sentiment/validation-300-predictions.jsonl

# All 589 validation sentences, without balanced subsampling.
uv run python -m sentiment_benchmark --compare-noul --all-validation \
  --max-calls 589 --model jev-1.13.0 \
  --output sentiment_benchmark/validation/comparison-validation.jsonl
```

A **tqdm progress bar** shows completed sentences, percentage, elapsed time, speed,
and ETA on stderr. Results and metrics are printed on stdout. Each sentence is
one sequential PydanticAI run with a one-model-request usage limit, a 30-second
HTTP timeout, and no agent retries. The native provider's underlying SDK uses
its default transport retry policy; network retries can add HTTP attempts beyond
`--max-calls`. That option limits agent requests, not total network attempts.

The full command above uses the recorded run's name, `comparison-validation`. Its output
files are already included in this folder; to rerun, choose a fresh filename such
as `comparison-validation-rerun.jsonl`. Existing outputs are never overwritten.
The full run is fresh: it does not reuse previous predictions. Actual charges
depend on current pricing and usage. `--max-calls` is **not a monetary cap**.
See [results.md](results.md) for measured costs and validation performance.

### CLI options

| Argument | Default | Purpose |
| --- | --- | --- |
| `--data PATH` | `data/sentiment/sentiment-labelled-sentences.zip` | Official ZIP cache path. |
| `--download` | Off | Download the archive if missing, then run inference. |
| `--source SOURCE` | `all` | `all`, `amazon_cells`, `imdb`, or `yelp`. |
| `--limit N` | `30` | Balanced sample size: multiple of 6 for all sources, or 2 for one. |
| `--all-validation` | Off | All validation rows for selected sources; incompatible with `--limit`. |
| `--seed N` | `42` | Sampling seed; ignored with `--all-validation`. |
| `--compare-noul` | Off | Compare both primitives; otherwise Choice only. |
| `--max-calls N` | `30` | Reject plans above this agent-request limit; transport retries can add HTTP attempts. |
| `--model ID` | `jev-1.13.0` | Jev ID without a `typesafe:` prefix. |
| `--output PATH` | Required | Fresh `.jsonl` path; a `.summary.json` report is also created. |
| `-h`, `--help` | — | Show help without inference. |

## Choice versus Noul

- **Choice:** `Prediction.sentiment` is a typed positive/negative enum.
- **Noul:** a float bounded by 0 and 1 asks whether the overall sentiment is positive.
  PydanticAI compiles it into a Noul and retains the raw positive probability.
- **Decision rule:** probability **≥ 0.5 → positive**, otherwise negative. The
  threshold is fixed in advance, not tuned on validation labels.

The two questions are evaluated independently in **one shared request per sentence**.
Neither sees the other's answer, gold labels, source, row ID, or previous predictions.
A Noul has no separate confidence; its probability is not sentiment intensity.
Paired mode keeps request count unchanged but adds tokens. Shared request latency
and usage do not measure either primitive's standalone cost or speed. The question
framing differs too, so results do not isolate primitive type perfectly.

## Results

See [results.md](results.md) for the original-versus-rerun comparison, per-source
metrics, prediction stability, representative failures, and measured costs.
Recorded predictions and reports are versionable files under [`validation/`](validation/).

## Reports and limitations

For `run.jsonl`, the runner also creates `run.summary.json`. Both paths must be fresh.

- **JSONL:** original text, ID/source/gold, Choice prediction, correctness, confidence,
  probabilities, resolved model, usage, and latency. Paired mode adds a `noul` object
  with positive probability, threshold, predicted label, and correctness.
- **Summary:** accuracy, macro-F1, per-label metrics, confusion matrices (gold rows,
  predicted columns), and per-source metrics. The `comparison` section adds Noul
  metrics, agreement, Noul-minus-Choice accuracy, paired correctness counts, and
  disagreement IDs. Top-level metrics continue to describe Choice.
- **Manifest:** dataset URL/hash, output schema/question descriptions, Python source
  hashes, dependency versions, requested model, split/sampling provenance, timestamp,
  and call budget. Decimal costs in usage metadata are serialized as exact strings.

Successful rows are flushed immediately. Failure stops the run with a nonzero exit
status and retains completed rows and a `partial` summary with attempted-run counts.
Partial metrics describe completed rows only. Hard termination may prevent the
summary from being written. There is no automatic resume or intentional overwrite.

This is a small, old, public corpus; pretraining contamination is possible. The
repository-defined validation split is not an official UCI protocol and cannot
rule out training exposure. Confidence is not correctness, and label agreement is
not universal sentiment truth. Do not tune prompts or thresholds on this holdout
if you intend to treat it as independent validation. Earlier full-corpus reports
must not be relabeled as validation results.
