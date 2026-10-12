from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .judge import AnswerJudge
from .manifest import (
    deployment_fingerprints,
    hash_file,
    hash_json,
    model_revisions,
    write_or_validate_manifest,
)
from .runner import JsonlWriter, latest_by_question


def build_judge_manifest(
    *,
    predictions_path: str | Path,
    inference_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    judge_mode: str,
) -> dict[str, Any]:
    judge_config = config["models"]["judge"]
    manifest = {
        "manifest_version": 2,
        "inference_run_fingerprint": inference_manifest["run_fingerprint"],
        "predictions_sha256": hash_file(predictions_path),
        "judge_mode": judge_mode,
        "judge_model": judge_config["model"],
        "judge_revision": model_revisions(config)["judge"],
        "judge_deployment_fingerprint": deployment_fingerprints(config)["judge"],
        "judge_configuration": judge_config,
        "rubrics_fingerprint": hash_json(config.get("judge_rubrics", {})),
    }
    manifest["judge_run_fingerprint"] = hash_json(manifest)
    return manifest


def run_judging(
    *,
    predictions_path: str | Path,
    judgments_path: str | Path,
    judge_manifest_path: str | Path,
    inference_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    judge_mode: str,
    retry_errors: bool = False,
    limit_questions: int = 0,
) -> dict[str, Any]:
    predictions = latest_by_question(predictions_path)
    manifest = build_judge_manifest(
        predictions_path=predictions_path,
        inference_manifest=inference_manifest,
        config=config,
        judge_mode=judge_mode,
    )
    write_or_validate_manifest(judge_manifest_path, manifest)
    existing = latest_by_question(judgments_path)
    mixed = [
        key
        for key, row in existing.items()
        if row.get("judge_run_fingerprint") != manifest["judge_run_fingerprint"]
    ]
    if mixed:
        raise ValueError(
            "Resume refused because existing judgments do not match "
            f"the current judge manifest ({len(mixed)} records)."
        )
    writer = JsonlWriter(judgments_path)
    judge = AnswerJudge(judge_mode, config["models"]["judge"])
    rubrics = config.get("judge_rubrics", {})
    processed = 0

    for key, prediction in predictions.items():
        previous = existing.get(key)
        if previous and not (retry_errors and previous.get("status") == "error"):
            continue
        if limit_questions and processed >= limit_questions:
            break

        if prediction.get("error"):
            result = {
                "is_correct": False,
                "reason": "Inference failure; counted as incorrect.",
                "mode": judge_mode,
                "status": "inference_error",
                "error": prediction["error"],
            }
        else:
            result = judge.judge(
                question=str(prediction["question"]),
                reference=str(prediction["reference_answer"]),
                prediction=str(prediction["prediction"]),
                task=str(prediction["task"]),
                rubric=str(
                    rubrics.get(
                        prediction["task"],
                        rubrics.get("default", ""),
                    )
                ),
            ).as_dict()
        row = {
            "question_key": key,
            "is_correct": result["is_correct"],
            "status": result["status"],
            "reason": result["reason"],
            "error": result.get("error"),
            "judge_mode": result["mode"],
            "judge_run_fingerprint": manifest["judge_run_fingerprint"],
        }
        writer.append(row)
        existing[key] = row
        processed += 1

    counts: dict[str, int] = {}
    for row in latest_by_question(judgments_path).values():
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "prediction_records": len(predictions),
        "judgment_records": len(existing),
        "processed_this_run": processed,
        "status_counts": counts,
        "is_fully_judged": len(existing) == len(predictions),
    }
