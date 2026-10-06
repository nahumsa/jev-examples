# Proof Practice quality evaluations

Uses **Pydantic Evals** (`pydantic-evals`, imported as `pydantic_evals`) to evaluate
the real application grading and comparison paths. No additional LLM judge is
used: Jev's rubric outputs are checked against versioned, hand-authored expectations.

## No-cost preview and offline checks

Run from `proof_practice/backend`:

```sh
uv sync --locked
uv run python -m evals --list
uv run pytest tests/test_evals.py -q
```

`--list` never constructs a provider or reads credentials. The initial dataset has
17 cases and 19 Jev calls per repeat (15 single-answer cases, 2 comparisons).
Offline tests use synthetic provider responses and exercise the actual PydanticAI
agent, comparison path, Pydantic Evals evaluators, reporting, and CLI safeguards.
They validate the harness, **not Jev's mathematical judgment**.

## Live evaluation (paid)

Set `TYPESAFE_API_KEY` in the backend `.env` or environment. Review current provider
pricing and the selected input sizes before authorizing inference. First run a
small sample, ideally with a pinned model:

```sh
uv run python -m evals --list --limit 3
uv run python -m evals --allow-paid --limit 3 --model jev-1.13.0 \
  --output eval-results/dev-smoke.json

# Entire starter dataset; a fresh output path is required.
uv run python -m evals --allow-paid --model jev-1.13.0 \
  --output eval-results/dev-full.json

# Repeated measurements of specific cases; 2 cases × 3 repeats = 9 calls here.
uv run python -m evals --allow-paid --model jev-1.13.0 \
  --case correct-even-sum --case comparison-revision-improves \
  --repeat 3 --concurrency 1 --output eval-results/dev-repeat.json
```

`--allow-paid` authorizes charges; it is **not a monetary cap**. Provider retries
are disabled in this runner and there are no LLM judges or automatic task retries.
The runner prints the maximum planned SDK invocation count before inference.
Concurrency defaults to one case at a time; a comparison invokes two calls in
parallel, so peak network concurrency can be twice `--concurrency`.

The app's shared 55-second per-answer timeout still applies. A failed task or
evaluator is recorded as a failure, never silently treated as a passing judgment.
There is no resume or per-case checkpoint support: interruption before the final
report is written may leave an empty file. Keep that file and choose a new path
for another run. Existing report files are never intentionally overwritten.

Exit status is 0 only if all assertions pass without task/evaluator failures;
1 means evaluation failure, and 2 means invalid CLI configuration. `--list` exits 0.
Do not run live evaluations in CI by default.

## Dataset and evaluators

[`cases.json`](cases.json) is loaded as
`Dataset[EvalInput, EvalOutput, Expectations]`. Metadata holds labels and rationale;
the task receives only the question and submitted proof(s). Expected labels and
score thresholds are never included in Jev state.

Coverage includes:

- Correct proofs across all six built-in questions, concise proofs, and alternative methods.
- Examples-only reasoning, circular arguments, false algebra, a missing case,
  and an induction proof with no base case.
- A custom √2 proof, a false custom claim, and prompt injection in both proof and question text.
- Revision quality and a comparison where polished circular prose must not outrank valid reasoning.

Three deterministic evaluators run for each applicable case:

| Evaluator | Checks |
| --- | --- |
| `OutputContract` | Exactly the five rubric dimensions, truthful total, original criteria, matching template feedback, correct question identity, and one/two SDK calls |
| `RubricScoreBands` | Each labeled dimension within its expected 0–2 interval; reports the mean distance outside those intervals (lower is better) |
| `ComparisonImprovement` | The intended better answer improves total and/or logical reasoning by a specified minimum; reports B-minus-A total |

Fractional Scores are stochastic, so assertions use intervals rather than exact
values. There is no minimum confidence threshold: concentration is not correctness.

**The labels are a small, hand-authored starter suite—not independent expert
annotations, a calibrated grading standard, or formal verification.** Have a
mathematics instructor review them, record disagreements, and adjust ranges with
justification. Use this suite for development; collect separate held-out student
proofs before making claims about grading accuracy. Do not repeatedly tune thresholds
to make the current model pass. Prompt-injection cases are probes, not a guarantee
of injection resistance.

## Reports and adding cases

JSON reports contain case inputs, both rubric outputs where applicable, assertions
with reasons, scores, failures, latency, actual call count, token usage, and resolved
model. Experiment metadata records requested model, dependency versions, dataset,
rubric and Python source SHA-256 hashes, repeat/concurrency settings, and start time. No credentials
are serialized. `eval-results/` is Git-ignored; reports can contain sensitive text
if you later add real student submissions. Review them before sharing.

Add a uniquely named entry to `cases.json` with:

1. `inputs.submission`: `problem_id` **or** `custom_statement`, plus `proof`.
2. Optional `inputs.second_proof` for a comparison.
3. `metadata`: category, rationale, and `answer_a` score bands keyed by rubric ID.
4. For comparisons: `answer_b` bands and optional minimum total/logic improvements.

Each band has `minimum` (default 0) and `maximum` (default 2). Only label dimensions
you can defend; the loader rejects unknown rubric IDs, unknown built-in questions,
missing labels, duplicate names, or comparison labels without a second answer.

Run the offline checks after changing cases or evaluators, then explicitly authorize
any live run. A passing report is evidence about this dataset, not a proof certificate.
