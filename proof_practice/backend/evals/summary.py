"""Render a saved Pydantic Evals report as a GitHub Actions Markdown summary."""

import argparse
import html
from pathlib import Path

from .__main__ import REPORT_ADAPTER, Report, report_passed


def cell(value: str) -> str:
    return html.escape(value).replace("|", "\\|").replace("\n", " ").replace("`", "\\`")


def render_summary(report: Report) -> str:
    metadata = report.experiment_metadata or {}
    assertion_count = sum(len(case.assertions) for case in report.cases)
    passing = sum(result.value for case in report.cases for result in case.assertions.values())
    calls = sum(case.metrics.get("jev_calls", 0) for case in report.cases)
    lines = [
        "## Proof Practice — live Jev evaluation",
        "",
        f"**Status:** {'PASS' if report_passed(report) else 'FAIL'}",
        f"**Requested model:** {cell(str(metadata.get('requested_model', 'unknown')))}",
        f"**Completed cases:** {len(report.cases)}; **task failures:** {len(report.failures)}",
        f"**Assertions:** {passing}/{assertion_count}; **SDK calls in completed cases:** {calls}",
        "",
        "| Case | Assertions passed | SDK calls | Duration |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case in report.cases:
        count = sum(result.value for result in case.assertions.values())
        lines.append(
            f"| {cell(case.name)} | {count}/{len(case.assertions)} | "
            f"{case.metrics.get('jev_calls', 0)} | {case.task_duration:.2f}s |"
        )
    for case in report.failures:
        lines.append(f"| {cell(case.name)} | Task failed | — | — |")
    failed = [
        (case.name, name)
        for case in report.cases
        for name, result in case.assertions.items()
        if not result.value
    ]
    if failed:
        lines.extend(["", "### Failed checks", ""])
        lines.extend(f"- {cell(case)}: {cell(name)}" for case, name in failed)
    evaluator_failures = sum(len(case.evaluator_failures) for case in report.cases)
    evaluator_failures += len(report.report_evaluator_failures)
    if evaluator_failures:
        lines.extend(["", f"**Evaluator failures:** {evaluator_failures}"])
    lines.extend(
        [
            "",
            "Download the JSON artifact for outputs, score differences, "
            "probabilities, and reasons.",
            "Task failures may have incurred calls not included in completed-case metrics.",
            "These are curated mathematical cases with provisional labels, not independently "
            "expert-validated student data or formal proof verification.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    report = REPORT_ADAPTER.validate_json(args.report.read_bytes())
    print(render_summary(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
