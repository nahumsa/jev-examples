# Sentiment benchmark results

## Runs and validation checks

Both runs on **2026-10-06** evaluated the same **589 validation sentences** using
`jev-1.13.0`, identical saved prompt schemas, and a fixed Noul threshold of **0.5**.
The holdout contains 296 negative and 293 positive sentences.

| Run | Client setup | Predictions | Report |
| --- | --- | --- | --- |
| Original: `comparison-validation` | PydanticAI with an explicitly configured SDK client; transport retries disabled | [JSONL](validation/comparison-validation.jsonl) | [Summary](validation/comparison-validation.summary.json) |
| Rerun: `comparison-validation-rerun` | PydanticAI-managed client; provider-default transport retries | [JSONL](validation/comparison-validation-rerun.jsonl) | [Summary](validation/comparison-validation-rerun.summary.json) |

Both runs completed successfully. Checks confirmed unique IDs, exact full
validation membership, original text and gold labels, correct threshold decisions,
and agreement between recomputed metrics and the saved reports. These artifacts
are kept alongside the code in `validation/` and are not Git-ignored. Dataset
caches and temporary outputs under root `data/` and `results/` remain ignored.

## Original versus rerun

| Metric | Original | Rerun |
| --- | ---: | ---: |
| Choice accuracy | **98.47% (580/589)** | **98.47% (580/589)** |
| Choice macro-F1 | 0.9847 | 0.9847 |
| Choice errors | 9 | 9 |
| Choice Brier score ↓ | 0.011969 | **0.011511** |
| Noul accuracy | **98.13% (578/589)** | 97.96% (577/589) |
| Noul macro-F1 | **0.9813** | 0.9796 |
| Noul errors | **11** | 12 |
| Noul Brier score ↓ | 0.020049 | **0.020023** |
| Choice–Noul agreement | **99.32% (585/589)** | 99.15% (584/589) |
| Reported cost | $0.013292202 | $0.013292202 |

Binary Brier scores are the mean of `(p_positive - gold_positive)^2`, calculated
from saved positive probabilities. Lower is better. These are supplementary
analysis, not fields currently emitted by the summary runner. A lower Brier score
can coexist with an additional classification error when a probability crosses
the fixed threshold; it does not by itself establish calibration on other data.

### Accuracy by source

| Source | Samples | Choice: original | Choice: rerun | Noul: original | Noul: rerun |
| --- | ---: | ---: | ---: | ---: | ---: |
| Amazon | 212 | 98.58% | 98.58% | 98.58% | 98.58% |
| IMDb | 183 | 97.81% | 97.81% | 97.27% | 97.27% |
| Yelp | 194 | 98.97% | 98.97% | 98.45% | 97.94% |

## Prediction stability

- **Choice:** all 589 predicted labels remained identical. Positive probabilities
  changed on 62 sentences.
- **Noul:** positive probabilities changed on 185 sentences, but only **one label
  flipped**, producing one additional error.
- **Changed sentence (`yelp:220`):** “- the food is rich so order accordingly.”
  UCI labels it positive. Noul predicted positive in the original run, then
  negative in the rerun with positive probability **0.45**. Choice remained positive.

Client setup changed between runs, but these two observations do not isolate
whether that change caused the variation. Identical prompt schemas and unchanged
Choice labels do not imply identical probability outputs or deterministic inference.

## Paired correctness and interpretation

| Outcome | Original | Rerun |
| --- | ---: | ---: |
| Both correct | 577 | 576 |
| Choice alone correct | 3 | 4 |
| Noul alone correct | 1 | 1 |
| Both wrong | 8 | 8 |
| Choice accuracy advantage | 0.34 percentage points | 0.51 percentage points |
| Exploratory exact two-sided McNemar p-value | 0.625 | 0.375 |

Choice wins numerically by two examples originally and three in the rerun.
Neither paired comparison provides convincing statistical evidence of a general
accuracy advantage. These runs repeat the same holdout; they are not two
independent validation datasets and should not be pooled as 1,178 distinct samples.

The original disagreement IDs were `imdb:326`, `imdb:737`, `imdb:949`, and
`yelp:472`. The rerun adds `yelp:220`. Noul alone is correct on `imdb:737`:
“That was done in the second movie.” UCI labels it negative, while Choice predicts
positive. The sentence's sentiment is difficult to interpret without context.

## Representative failures

| Sentence | UCI label | Observed result |
| --- | --- | --- |
| “It definitely was not as good as my S11.” | Positive | Both predict negative in both runs. In the original run, Choice confidence is 1.0 and Noul positive probability is 0.05. The label appears questionable from the sentence alone. |
| “Bacon is hella salty.” | Positive | Both predict negative in both runs. Interpretation may depend on missing context or personal preferences. |
| “You can find better movies at youtube.” | Negative | Both predict positive in both runs, missing the comparative criticism. |
| “Think of the film being like a dream.” | Positive | Choice predicts positive in both runs; Noul returns 0.46 and predicts negative in both. |

These examples suggest a mixture of genuine model errors, context-dependent
sentiment, and possible label inconsistencies. Labels were not changed after
inspection, and the threshold was not adjusted to improve the validation score.
Confidence is not a correctness guarantee.

## Measured cost

Each run recorded **589 model requests**, **316,481 input tokens**, and **28,861
output tokens**. Summed usage metadata reports **$0.013292202 per run**, about
**1.33 cents**, matching $0.042 per million input tokens with output tokens free
at that published rate.

The two full runs together report **$0.026584404**, about **2.66 cents**. This does
not include earlier samples, failed attempts, or other account usage. Logged model
requests do not expose every possible transport retry attempt. Choice and Noul
share requests, so these costs are not separate per-primitive cost measurements.

## Takeaway

Both primitives perform strongly on this small public holdout. Choice has a
narrow numerical edge, lower Brier scores, and stable labels across these two runs.
Noul's probability variation produced one additional threshold-crossing error.
The results are very similar and do not establish a reliable general winner.

The validation split is repository-defined, not an official UCI protocol. This
old public corpus may have appeared in pretraining; a local holdout cannot rule
that out. Do not tune prompts or thresholds on this holdout if you intend to treat
it as independent validation. These results do not establish production readiness.
