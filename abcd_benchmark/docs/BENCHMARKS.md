# Jev on the ABCD paper tasks

`python -m abcd_benchmark` runs **Action State Tracking (AST)** and **Cascading
Dialogue Success (CDS)**. `python -m abcd_benchmark.cli.intent` runs the separate
conversation-prefix intent classifier. Run all commands from the repository root.

Sources:
- [Paper, sections 6 and Appendix C](https://aclanthology.org/2021.naacl-main.239/)
- [Reference preprocessing](https://github.com/asappresearch/abcd/blob/6b8700ce67c6b37b062dd7a60abc76d7ef832a97/utils/process.py)
- [Reference scoring](https://github.com/asappresearch/abcd/blob/6b8700ce67c6b37b062dd7a60abc76d7ef832a97/utils/evaluate.py)

## Run

```bash
uv sync
# Requests only; may download public data and the pinned BERT vocabulary:
uv run python -m abcd_benchmark --task ast --split dev --limit 1 --dry-run \
  --output results/ast-inputs.jsonl
uv run python -m abcd_benchmark --task cds --split dev --limit 1 --dry-run \
  --output results/cds-inputs.jsonl

# Paid examples: approve cost/budget first; requires TYPESAFE_API_KEY.
uv run python -m abcd_benchmark --task ast --split dev --limit 3 \
  --output results/ast-dev-3.jsonl
uv run python -m abcd_benchmark --task cds --split dev --limit 3 \
  --output results/cds-dev-3.jsonl

# Fixed full-test evaluations, only after budget approval and dev validation:
uv run python -m abcd_benchmark --task ast --split test --full-split \
  --output results/ast-test-full.jsonl
uv run python -m abcd_benchmark --task cds --split test --full-split \
  --output results/cds-test-full.jsonl

uv run pytest -q
```

The default model is pinned to `jev-1.13.0`. `--model` takes a TypeSafe model ID,
not Pydantic AI's `typesafe:` prefix. `--concurrency 4` bounds **network requests**,
not just examples. `--limit` counts conversations; `--example-limit` is a debugging
cap and disables cascading scores if it truncates the selected conversations.
Defaults are three sampled dev conversations with seed 42. Outputs must be new.
The SDK retries transient errors; requests have a 60-second HTTP timeout.

## Protocol

Protocol ID: `abcd-v1.1-reference-bert-512`.

The implementation follows the pinned dataset and reference BERT preprocessing,
without training a model. The BERT tokenizer vocabulary is pinned to Hugging Face
revision `86b5e0934494bd15c9632b12f734a8a67f723594`. Only vocabulary files are
downloaded, not model weights. Ontology entity placeholders are added as tokens.

- Predict **before** each action (AST) or agent/action turn (CDS).
- Current/future turns and scenario metadata are never model inputs.
- Prior action names enter AST history; prior action result text enters CDS
  history, matching the reference's different history construction.
- Speakers are removed, turns joined with `[SEP]`, and the **first 510 BERT
  tokens** retained, not the most recent tail. Jev sees those tokens as text.
- Copy context uses unique tokens longer than two characters and retains the
  reference's bounded tail. Enumerable value IDs preserve ontology ordering.
- AST omits value-bearing examples whose reference target cannot be resolved.
  CDS retains them with value target `-1`.
- `verify-identity` and `validate-purchase` expand into three slot examples.
  Stable ordering preserves duplicate turn counts. CDS does not expose the slot
  position, matching the upstream implementation.
- CDS adds a terminal example using the final turn count, even when duplicated.
- Retrieval uses each example's exact 100 supplied candidate IDs and
  `utterances.json`, preserving duplicates and ordering. Candidate text is
  truncated to 510 BERT tokens, as in the reference encoder.

### Jev judgments

The SDK returns typed Choice answers with distributions and confidence:

1. **History-only request:** action; also intent and next step for CDS.
2. **Value request:** history plus the official copy context, choosing among
   enumerable values and context-token positions. It runs for every example,
   including examples without applicable value labels.
3. **Retrieval request:** when the benchmark supplies a candidate pool, choose
   among the 100 responses; probabilities provide Recall@1/5/10 rankings.

**Separate states matter:** candidate-pool presence and copy-context presence can
reveal the annotated next-step branch. They are never included in the history-only
request. Questions cannot read answers from other requests. The value head infers
its action premise from history; no gold action/intent is explicitly provided.

There is no oracle-intent input, scenario input, guideline injection, or KB masking.
The action/value/response realizations are speculative heads; the scorer consumes
only the realization required by the gold branch, as the official evaluator does.

## Metrics

AST:
- `Bslot_Accuracy`: action correct / all retained AST examples.
- `Value_Accuracy`: value correct / all retained AST examples.
- `Joint_Accuracy`: action and value correct / all retained AST examples.

CDS:
- `Intent_Accuracy`, `Nextstep_Accuracy`: all decision examples.
- `Action_Accuracy`, `Joint_Accuracy`: examples with action labels >= 0.
- `Value_Accuracy`: examples with value labels >= 0.
- `Recall_at_1`, `Recall_at_5`, `Recall_at_10`: supplied retrieval examples.
- `Turn_Accuracy`: intent and next step correct, plus top-1 retrieval or joint
  action/value correctness for the applicable branch. Ending needs no realization.
- `Cascading_Score`: for every example start, consecutive correct examples until
  the first error divided by remaining examples; average over all example starts.
  Example correctness `[true,true,false,true]` scores `(2/4 + 1/3 + 0 + 1)/4`.

Metrics are computed offline on **recorded histories**, not on model-generated
rollouts. Conversations are weighted by their number of decision examples.
`numpy.argpartition` preserves the reference's retrieval tie behavior. Zero
support returns `null` rather than dividing by zero. Partial/truncated CDS runs
have `Cascading_Score: null`, since remaining-length denominators are incomplete.

### Reference quirks — deliberately not corrected

1. A value target of `-1` never equals a nonnegative argmax prediction. Consequently,
   **no-value actions fail joint accuracy and CDS turn correctness**. AST also
   includes these examples in its value-accuracy denominator. This is how the
   pinned evaluator behaves, even though a semantic evaluator could treat absence
   of a required value as success. Summary JSON reports their count explicitly.
2. For non-enumerable values, the first matching category's placeholder is used,
   rather than checking the literal annotated value. Expanded CDS slots may thus
   share the same value target and the same request.
3. Copy-context length uses the annotated action's token length; this inherited
   preprocessing artifact remains, although its name is not passed to Jev.
4. The reference CDS code omits slot positions in action/value feature construction.

These choices make labels and scoring comparable to the pinned reference code,
not a corrected end-to-end business-process simulator. The paper trained several
models/tokenizers and averaged three seeds; a single zero-shot Jev evaluation on
v1.1 is **not an exact reproduction of those experimental conditions**.

## Raw-input experiment (no reference tokenizer)

```bash
# Paid runs: approve cost/budget before executing.
uv run python -m abcd_benchmark --task ast --split test --full-split --preprocessing raw \
  --output results/ast-test-raw.jsonl
uv run python -m abcd_benchmark --task cds --split test --full-split --preprocessing raw \
  --output results/cds-test-raw.jsonl
```

Protocol: `abcd-v1.1-raw-full-history-entities`. This mode never loads a BERT
vocabulary or invokes a reference tokenizer. It sends the full observed dialogue
text with line-separated turns and preserves the full response candidate text.
"Raw" refers to the supplied ABCD **delexed** text, not its separate `original`
field: entity placeholders remain as provided by the dataset.

Value choices consist of the same ontology values plus distinct entity
placeholders actually observed in history. No WordPiece fragments or token-position
tail truncation are used. Entity-choice construction is independent of gold action;
only gold target calculation uses annotation metadata outside the requests.

Action histories, speaker omission, decision points, multi-slot expansion,
missing-value scoring quirks, model, and SDK judgment implementation remain the
same. This is a **preprocessing ablation**, not the exact reference benchmark:
fuller history and fewer copy choices can change task difficulty and eligible
examples. Compare semantic value strings rather than copy-position IDs across
protocols, and use common example IDs if the eligible cohorts differ. Summaries
record `tokenizer: null` and the distinct protocol ID.

## Artifacts

Every JSONL row contains:
- Conversation, turn, slot, terminal flag, and unique example ID.
- Gold IDs (outside model input), predicted IDs, and readable `output`.
- Per-question `score` (Choice confidence), full distributions, and correctness.
- Exact `requests`: states, instructions, and criteria, making leakage auditable.
- Resolved model IDs, token usage, successful request count, and latency.

Adjacent `.summary.json` contains full-precision and reference-rounded metrics,
metric denominators, ID mappings, all confusion pairs, confidence, token usage,
cost estimate, completion status, selected/full-split scope, failed example IDs,
protocol notes, dataset revision, tokenizer revision, and input SHA-256 hashes.
Copy-value IDs above the enumerable range refer to **example-local positions**;
use the row's value-request criteria to interpret them.

Successful examples are flushed even if another example fails. A partial summary
is written and the error propagated; there is no automatic resume. Estimated cost
uses $0.042 per million input tokens with free output; it excludes retries and
successful heads belonging to failed examples and is not an invoice.

## Validation performed

- Pytest executes relevant definitions from unmodified pinned upstream source
  fixtures, with only tensor/embedding plumbing replaced. It compares example
  expansion, context tokens, labels, and all metrics, including ties and duplicate
  turn counts. Fixtures include the upstream MIT license.
- Local full-test preprocessing parity was checked against upstream for all 1,004
  conversations: **4,562 AST examples**, **15,368 CDS examples**. This validation
  made no model calls (`results/reference-preprocessing-validation.json`).
- Real one-conversation dev smoke runs completed for both tasks:
  `results/ast-dev-1.jsonl` and `results/cds-dev-1.jsonl`, with adjacent summaries.
  These small runs validate the pipeline; they are not full benchmark results.
