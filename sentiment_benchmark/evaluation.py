"""Bounded PydanticAI runs, flushed artifacts, and deterministic label metrics."""

import json
from collections import Counter
from pathlib import Path

from pydantic_ai import Agent
from tqdm.auto import tqdm

from .data import LABELS, Sentence
from .model import NOUL_THRESHOLD, ComparisonPrediction, Prediction, predict


def metrics(records: list[dict]) -> dict:
    matrix = {gold: {label: 0 for label in LABELS} for gold in LABELS}
    for row in records:
        matrix[row["gold"]][row["choice"]] += 1
    correct = sum(matrix[label][label] for label in LABELS)
    per_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        per_label[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0,
        }
    return {
        "count": len(records),
        "accuracy": correct / len(records) if records else None,
        "macro_f1": sum(r["f1"] for r in per_label.values()) / 2 if records else None,
        "confusion_matrix": matrix,
        "per_label": per_label,
    }


def comparison_metrics(records: list[dict]) -> dict:
    noul_rows = [{**row, "choice": row["noul"]["choice"]} for row in records]
    choice_metrics = metrics(records)
    noul_metrics = metrics(noul_rows)
    paired = {
        "both_correct": 0,
        "choice_only_correct": 0,
        "noul_only_correct": 0,
        "both_wrong": 0,
    }
    disagreement_ids = []
    for row in records:
        choice_correct = row["choice"] == row["gold"]
        noul_correct = row["noul"]["choice"] == row["gold"]
        if choice_correct and noul_correct:
            paired["both_correct"] += 1
        elif choice_correct:
            paired["choice_only_correct"] += 1
        elif noul_correct:
            paired["noul_only_correct"] += 1
        else:
            paired["both_wrong"] += 1
        if row["choice"] != row["noul"]["choice"]:
            disagreement_ids.append(row["id"])
    return {
        "noul_threshold": NOUL_THRESHOLD,
        "choice": choice_metrics,
        "noul": noul_metrics,
        "noul_minus_choice_accuracy": (
            noul_metrics["accuracy"] - choice_metrics["accuracy"] if records else None
        ),
        "agreement_rate": 1 - len(disagreement_ids) / len(records) if records else None,
        "paired_correctness": paired,
        "disagreement_ids": disagreement_ids,
        "per_source_noul": {
            source: metrics([row for row in noul_rows if row["source"] == source])
            for source in sorted({row["source"] for row in records})
        },
    }


async def run(
    rows: list[Sentence],
    *,
    output: Path,
    manifest: dict,
    agent: Agent,
) -> dict:
    summary_path = output.with_suffix(".summary.json")
    if output == summary_path or output.exists() or summary_path.exists():
        raise ValueError(
            "Output/report already exists or paths conflict; choose a fresh .jsonl path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    status = "partial"
    attempts = 0
    with (
        output.open("x", encoding="utf-8") as predictions,
        summary_path.open("x", encoding="utf-8") as summary_file,
    ):
        try:
            for sentence in tqdm(rows, desc="Validation inference", unit="sentence"):
                attempts += 1
                record = await predict(agent, sentence)
                predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                predictions.flush()
                completed.append(record)
            status = "complete"
        finally:
            sources = sorted({row.source for row in rows})
            summary = {
                "status": status,
                "manifest": manifest,
                "selected_count": len(rows),
                "completed_count": len(completed),
                "selected_ids": [row.id for row in rows],
                "agent_runs_attempted": attempts,
                "label_counts": dict(Counter(row.gold for row in rows)),
                "output_schema": manifest.get(
                    "output_schema",
                    (
                        ComparisonPrediction
                        if manifest.get("compare_noul")
                        else Prediction
                    ).model_json_schema(),
                ),
                "comparison": comparison_metrics(completed)
                if manifest.get("compare_noul")
                else None,
                "metrics": metrics(completed),
                "per_source": {
                    source: metrics([r for r in completed if r["source"] == source])
                    for source in sources
                },
            }
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
            summary_file.write("\n")
    return summary
