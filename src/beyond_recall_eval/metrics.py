from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evidence import EvidenceManifest
from .models import Trajectory
from .runner import latest_by_question


def wilson_interval(
    correct: int,
    total: int,
    z: float = 1.959963984540054,
) -> list[float] | None:
    if total == 0:
        return None
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def expected_rows(trajectories: Iterable[Trajectory]) -> list[dict[str, Any]]:
    return [
        {
            "question_key": f"{trajectory.key}/q{question.index}",
            "dimension": trajectory.dimension,
            "task": trajectory.task,
        }
        for trajectory in trajectories
        for question in trajectory.questions
    ]


def score_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(rows)
    predictions = sum(row["prediction_present"] for row in rows)
    judged = sum(row["judged"] for row in rows)
    unscored = sum(row["unscored"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    ready = predictions == expected and judged == expected
    return {
        "expected_questions": expected,
        "completed_predictions": predictions,
        "missing_predictions": expected - predictions,
        "prediction_coverage": predictions / expected if expected else 0,
        "judged_questions": judged,
        "unscored_questions": unscored,
        "correct": correct,
        "is_complete": ready,
        "strict_accuracy": correct / expected if ready and expected else None,
        "strict_wilson_95_ci": wilson_interval(correct, expected) if ready else None,
        "strict_accuracy_lower_bound": correct / expected if expected else None,
        "accuracy_on_judged_questions": correct / judged if judged else None,
        "inference_errors": sum(row["inference_error"] for row in rows),
        "judge_errors": sum(row["judge_error"] for row in rows),
    }


def grouped(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: score_block(value) for key, value in sorted(groups.items())}


def validate_fingerprints(
    predictions: Mapping[str, dict[str, Any]],
    judgments: Mapping[str, dict[str, Any]],
    inference_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any] | None,
) -> None:
    expected_inference = inference_manifest["run_fingerprint"]
    if any(
        row.get("run_fingerprint") != expected_inference for row in predictions.values()
    ):
        raise ValueError("Predictions contain mixed inference run fingerprints.")
    if judgments:
        if judge_manifest is None:
            raise ValueError("Judgments require a judge run manifest.")
        expected_judge = judge_manifest["judge_run_fingerprint"]
        if any(
            row.get("judge_run_fingerprint") != expected_judge
            for row in judgments.values()
        ):
            raise ValueError("Judgments contain mixed judge run fingerprints.")


def evidence_access_metrics(
    rows: list[dict[str, Any]],
    predictions: Mapping[str, dict[str, Any]],
    manifest: EvidenceManifest | None,
    top_k: int | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    evaluated = []
    row_by_key = {row["question_key"]: row for row in rows}
    for key, entry in manifest.entries.items():
        prediction = predictions.get(key)
        accessed = (
            set(str(item) for item in prediction.get("accessed_units", []))
            if prediction
            else set()
        )
        media_required = any(
            selector.startswith(("images/", "videos/")) for selector in entry.selectors
        )
        evaluated.append((row_by_key[key], entry.unit_ids, accessed, media_required))

    any_hit = sum(bool(required & accessed) for _, required, accessed, _ in evaluated)
    complete = [
        row for row, required, accessed, _ in evaluated if required.issubset(accessed)
    ]
    complete_correct = sum(row["correct"] for row in complete)
    total = len(evaluated)
    media_rows = [
        (row, required, accessed)
        for row, required, accessed, media_required in evaluated
        if media_required
    ]
    media_complete = [
        row for row, required, accessed in media_rows if required.issubset(accessed)
    ]
    return {
        "scope": "validated structured retrieval or exposure trace",
        "top_k": top_k,
        "manifest_expected_questions": manifest.expected_questions,
        "manifest_annotated_questions": manifest.annotated_questions,
        "manifest_coverage": (
            manifest.annotated_questions / manifest.expected_questions
            if manifest.expected_questions
            else 0
        ),
        "any_hit": any_hit,
        "any_hit_rate": any_hit / total if total else None,
        "complete_access": len(complete),
        "cesa": len(complete) / total if total else None,
        "correct_given_complete_access": complete_correct,
        "accuracy_given_cesa": (complete_correct / len(complete) if complete else None),
        "post_access_failure": (
            1 - complete_correct / len(complete) if complete else None
        ),
        "media_required_subset": {
            "questions": len(media_rows),
            "correct": sum(row["correct"] for row, _, _ in media_rows),
            "accuracy": (
                sum(row["correct"] for row, _, _ in media_rows) / len(media_rows)
                if media_rows
                else None
            ),
            "complete_access": len(media_complete),
            "cesa": (len(media_complete) / len(media_rows) if media_rows else None),
            "correct_given_complete_access": sum(
                row["correct"] for row in media_complete
            ),
            "accuracy_given_cesa": (
                sum(row["correct"] for row in media_complete) / len(media_complete)
                if media_complete
                else None
            ),
        },
    }


def summarize(
    *,
    trajectories: list[Trajectory],
    predictions_path: str | Path,
    judgments_path: str | Path | None,
    inference_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any] | None,
    evidence_manifest: EvidenceManifest | None,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    expected = expected_rows(trajectories)
    predictions = latest_by_question(predictions_path)
    judgments = latest_by_question(judgments_path) if judgments_path else {}
    expected_keys = {row['question_key'] for row in expected}
    if (set(predictions) | set(judgments)) - expected_keys:
        raise ValueError('Results contain questions outside the selected evaluation scope.')
    if judge_manifest and judge_manifest['inference_run_fingerprint'] != inference_manifest['run_fingerprint']:
        raise ValueError('Judgments belong to a different inference run.')
    validate_fingerprints(
        predictions,
        judgments,
        inference_manifest,
        judge_manifest,
    )

    rows = []
    for item in expected:
        key = item["question_key"]
        prediction = predictions.get(key)
        judgment = judgments.get(key)
        is_correct = judgment.get("is_correct") if judgment else None
        status = str(judgment.get("status", "")) if judgment else ""
        rows.append(
            {
                **item,
                "prediction_present": prediction is not None,
                "inference_error": bool(prediction and prediction.get("error")),
                "judged": isinstance(is_correct, bool),
                "unscored": status == "unscored" or judgment is None,
                "judge_error": status == "error",
                "correct": is_correct is True,
            }
        )

    by_task = grouped(rows, "task")
    task_scores = [
        block["strict_accuracy"]
        for block in by_task.values()
        if block["strict_accuracy"] is not None
    ]
    top_k = profile.get("retrieval_top_k")
    top_k = int(top_k) if top_k is not None else None
    return {
        "overall": score_block(rows),
        "by_dimension": grouped(rows, "dimension"),
        "by_task": by_task,
        "task_macro_accuracy": (
            sum(task_scores) / len(task_scores)
            if len(task_scores) == len(by_task) and task_scores
            else None
        ),
        "evidence_access": evidence_access_metrics(
            rows,
            predictions,
            evidence_manifest,
            top_k,
        ),
        "evaluation_scope": "representative_subset" if inference_manifest.get("sample_manifest_sha256") else "full_dataset",
        "selected_scope_complete": score_block(rows)["is_complete"],
        "formal_result_ready": score_block(rows)["is_complete"] and not inference_manifest.get("sample_manifest_sha256"),
    }


def write_summary(output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    summary = summarize(**kwargs)
    Path(output_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
