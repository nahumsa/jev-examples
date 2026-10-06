# Jev examples

## Proof Practice study app

[`proof_practice/`](proof_practice/README.md) is a standalone Next.js study workspace
with a FastAPI/PydanticAI backend and Jev rubric evaluation.

| Study feature | Implementation |
| --- | --- |
| Built-in questions and trusted reference proofs | [`problems.py`](proof_practice/backend/app/problems.py) |
| Five-part proof rubric | [`rubric.json`](proof_practice/backend/app/rubric.json) |
| Typed Jev grading workflow | [`grading.py`](proof_practice/backend/app/grading.py) |
| Custom questions and independent two-answer comparison API | [`main.py`](proof_practice/backend/app/main.py) |
| Proof editor, browser-saved questions, and side-by-side feedback | [`page.tsx`](proof_practice/frontend/app/page.tsx) |
| Local container deployment | [`docker-compose.yml`](proof_practice/docker-compose.yml) |
| API and grading regression tests | [`test_app.py`](proof_practice/backend/tests/test_app.py) |
| Pydantic Evals quality dataset, rubric checks, and paid-run safeguards | [`evals/`](proof_practice/backend/evals/README.md) |

For installation, API-key setup, Docker commands, and checks, see the
[Proof Practice README](proof_practice/README.md). The app has its own dependencies;
run its commands from `proof_practice/backend` or `proof_practice/frontend`, not
the repository root. Keep real `.env` files private; only placeholder templates belong in Git.

Jev scores are study feedback, **not formal proof verification**. Lean checking
and LLM-to-Lean formalization were discussed as future extensions and are not
implemented. See [verification scope](proof_practice/README.md#formal-verification-scope).

## Continuous integration

[Tests](.github/workflows/ci.yml) runs offline benchmark and Proof Practice tests,
backend linting, and the Next.js build on pushes and pull requests. It never uses
an API key or performs paid inference.

[Live Proof Practice evals](.github/workflows/proof-live-evals.yml) is a separate,
manually authorized workflow that evaluates curated proofs with real Jev and saves
per-case reports. See [secret setup, call budgets, and execution instructions](proof_practice/backend/evals/README.md#github-actions).

## Benchmarks

| Benchmark | Tasks and implementation | Documentation |
| --- | --- | --- |
| ABCD | [AST/CDS](abcd_benchmark/ast_cds/), [conversation-prefix intent routing](abcd_benchmark/intent/), and [offline tests](abcd_benchmark/tests/) | [Setup and protocol](abcd_benchmark/README.md) |
| Bitext | [Single-message intent classification](bitext_benchmark/evaluation.py) using Jev through PydanticAI, with [offline tests](bitext_benchmark/tests/) | [Setup and protocol](bitext_benchmark/README.md) |

Both benchmarks share the root Python environment. Proof Practice uses separate
backend and frontend environments.

Prepare a small Bitext run without inference:

```bash
uv run python -m bitext_benchmark --download --dry-run \
  --output results/bitext/dev-inputs.jsonl
```

From the repository root:

```bash
uv sync
uv run python -m abcd_benchmark --help
uv run python -m abcd_benchmark.cli.intent --help
uv run python -m abcd_benchmark.cli.download --help
uv run python -m abcd_benchmark.cli.summarize_cds --help
uv run pytest -q
```

Cached datasets remain in `data/abcd/` and prediction artifacts in `results/`.
Inference can incur charges; approve a cost estimate before running it.
