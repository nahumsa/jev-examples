# Modular code guidelines

Group modules by the feature they serve. Within each feature, separate independent
responsibilities. More files alone do not make code more modular.

## Folder ownership

```text
abcd_benchmark/
  ast_cds/       Recorded-turn AST/CDS evaluation
  intent/        Conversation-prefix intent routing
  common/        Shared offline dataset and artifact utilities
  cli/           Argument parsing and command entry points
  docs/          Protocol and architecture documentation
  tests/
    ast_cds/     Benchmark tests and reference fixtures
    intent/      Intent-routing tests
```

`ast_cds/` owns schema, config, preprocessing, tokenizer loading, example
construction, dataset preparation, requests, normalized contracts, inference,
prediction decoding, metrics, reporting, and execution. `pipeline.py` composes
inference and decoding; it also exports the established request helpers.

`intent/` owns its configuration, output enum, requests, agent factory, inference,
datasets, metrics, reporting, and runner. Historical adapter functions live in
`evaluation.py`, `model.py`, and `summary.py` inside that folder.

`common/` contains `data.py`, `selection.py`, and `artifacts.py`. These have no
inference dependencies. CLI commands live in `cli/`; the package's `__main__.py`
only forwards to the AST/CDS CLI.

## 1. Prefer cohesive folders and modules

Keep related features together. Split a module when responsibilities change
independently, not at an arbitrary line count. Avoid a new file or class for every
small helper. Do not build a generic framework for hypothetical future needs.

## 2. Make contracts explicit

Use typed functions, dataclasses, and small protocols. Pass configuration and
required dependencies explicitly instead of relying on mutable globals. Normalize
SDK responses before handing them to prediction decoding.

## 3. Keep calculations offline

Example construction, decoding, scoring, and summary assembly must work without
credentials or model calls. Put network/filesystem work behind explicit boundaries.
Request building has no file/network I/O; its optional candidate cache is
caller-owned mutable state. SDK `Choice` objects are still used to preserve request
schemas, so the request interface is not fully provider-neutral.

## 4. Keep orchestration thin

Runners coordinate loaders, requests, inference, output writing, and reporting.
They should not redefine sampling, prompts, scoring, or data transformation rules.
CLIs parse arguments and call a runner; they should not contain model logic.

## 5. Distinguish relocation from behavioral changes

Folder moves require updated import paths and commands, not changed benchmark
semantics. Preserve prompts, enum descriptions, IDs, probability order, tie rules,
and JSON fields. Backend migrations or new guideline prompts are separate changes.
The current reorganization does not change model interfaces or scoring behavior.

## 6. Protect information boundaries

Never send gold labels, future turns, or scenario metadata to the model.
Action/intent/next-step heads must not receive support inputs whose presence
reveals the annotated branch. Future action-conditioned value strategies must use
predicted actions, never annotated actions, and have their own protocol metadata.

## 7. Version experiments explicitly

Reference quirks are part of the evaluation protocol. New prompts, guidelines,
value conditioning, or candidate filters need explicit configuration, tests, and
separate experiment IDs. Do not silently change the reference pipeline.

## 8. Test features and boundaries

Use pytest, synthetic data, and fake clients. Cover leakage, raw/reference
preprocessing, copy values, ranking ties, missing labels, concurrency, failed heads,
partial summaries, exact completed-conversation coverage, and output guards.
Keep upstream parity fixtures. Regular tests must not make paid API calls.

## 9. Keep ownership and failures explicit

The runner owns client lifetime and output artifacts. Inference must wait for
sibling heads before reporting a failure. Preserve successful rows and report
partial results honestly. A terminal row alone does not prove completeness.

## 10. Approve cost before paid work

Count examples and requests offline, estimate tokens/cost with retry assumptions,
and obtain approval for a hard budget before running inference. Never
implicitly restart a paid run. Successful saved-call usage is not a complete
billing ledger: retries, failed heads, and in-flight calls can incur charges.

**Hard monetary caps and resume support are not yet implemented.** They are
follow-up work, not capabilities of this folder reorganization. Cached datasets
and prediction artifacts remain outside the code folder.
