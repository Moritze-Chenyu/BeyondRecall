from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .configuration import expected_runtime_info, validate_runtime_info
from .evidence import EvidenceManifest
from .interface import TrajectorySystem, normalize_prediction
from .models import RetrievalTraceItem, Trajectory, TrajectoryInfo
from .rendering import CaptionStore, InputBuilder
from .temporal import visible_history


CreateSystem = Callable[[TrajectoryInfo, Mapping[str, Any]], TrajectorySystem]


def safe_error(error: Exception, dataset_root: Path) -> str:
    text = f"{type(error).__name__}: {error}"
    text = text.replace(str(dataset_root), "<DATASET_ROOT>")
    text = re.sub(r"(?:/home|/Users|/data[^/\\s]*)/[^\\s:]+", "<LOCAL_PATH>", text)
    return text[:2000]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_by_question(path: str | Path) -> dict[str, dict[str, Any]]:
    latest = {}
    for record in load_jsonl(path):
        latest[str(record["question_key"])] = record
    return latest


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def question_key(trajectory: Trajectory, question_index: int) -> str:
    return f"{trajectory.key}/q{question_index}"


def namespace(
    run_id: str,
    profile_name: str,
    trajectory_key: str,
    question_index: int,
) -> str:
    raw = f"{run_id}|{profile_name}|{trajectory_key}|q{question_index}"
    return f"br_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def validate_trace(
    trace: tuple[RetrievalTraceItem, ...],
    valid_units: set[str],
    top_k: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if top_k is not None and len(trace) > top_k:
        raise ValueError(f"Retrieval trace has {len(trace)} rows but Top-K is {top_k}.")
    ranks = [item.rank for item in trace]
    if ranks != list(range(1, len(trace) + 1)):
        raise ValueError("Retrieval trace ranks must be contiguous and start at 1.")

    serialized = []
    accessed = []
    for item in trace:
        unknown = set(item.source_units) - valid_units
        if unknown:
            raise ValueError(
                f"Retrieval trace contains unknown source units: {unknown}"
            )
        serialized.append(
            {
                "rank": item.rank,
                "source_units": list(item.source_units),
                "delivered_to_answer": item.delivered_to_answer,
                "score": item.score,
            }
        )
        if item.delivered_to_answer:
            accessed.extend(item.source_units)
    return serialized, list(dict.fromkeys(accessed))


def run_inference(
    *,
    dataset_root: str | Path,
    trajectories: list[Trajectory],
    config: Mapping[str, Any],
    profile_name: str,
    profile: Mapping[str, Any],
    create_system: CreateSystem,
    output_path: str | Path,
    run_manifest: Mapping[str, Any],
    framework_version: str,
    caption_store_path: str | Path | None = None,
    evidence_manifest: EvidenceManifest | None = None,
    limit_questions: int = 0,
    retry_errors: bool = False,
    question_keys: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    writer = JsonlWriter(output_path)
    existing = latest_by_question(output_path)
    expected_fingerprint = run_manifest["run_fingerprint"]
    mixed = [
        key
        for key, row in existing.items()
        if row.get("run_fingerprint") != expected_fingerprint
    ]
    if mixed:
        raise ValueError(
            "Resume refused because existing predictions do not match "
            f"the current run manifest ({len(mixed)} records)."
        )
    processed = 0
    shared_caption_store = CaptionStore(caption_store_path)
    staging_root = Path(output_path).parent / ".media_staging"
    expected_runtime = expected_runtime_info(config, profile, framework_version)
    top_k = profile.get("retrieval_top_k")
    top_k = int(top_k) if top_k is not None else None
    adapter_media_processing = dict(config.get("media_processing", {}))
    adapter_media_processing["quality"] = {
        key: value for key, value in adapter_media_processing.get("quality", {}).items()
        if key not in {"pdf_fixed_pages", "pdf_evidence_pages", "pdf_crop_conflicts", "pdf_crop_invalid_annotations",
                       "video_segment_manifest", "video_segment_manifest_sha256"}
    }
    adapter_config = {
        "profile_name": profile_name,
        "profile": profile,
        "models": config["models"],
        "protocol": config["protocol"],
        "retrieval_defaults": config.get("retrieval_defaults", {}),
        "media_processing": adapter_media_processing,
        "prompts": config.get("prompts", {}),
        "framework_version": framework_version,
        "runtime_expectations": expected_runtime,
    }

    for trajectory in trajectories:
        for question in trajectory.questions:
            key = question_key(trajectory, question.index)
            if question_keys is not None and key not in question_keys:
                continue
            previous = existing.get(key)
            if previous and not (retry_errors and previous.get("error")):
                continue
            if limit_questions and processed >= limit_questions:
                break

            if profile.get("history_scope") == "private_annotated_evidence_only":
                if evidence_manifest is None:
                    raise ValueError(
                        "Oracle profiles require a private evidence manifest."
                    )
                source_sessions = evidence_manifest.oracle_sessions(
                    trajectory,
                    question.index,
                )
            elif profile.get("history_scope") == "question_only":
                source_sessions = ()
            elif profile.get("history_scope") == "all_sessions":
                source_sessions = trajectory.sessions
            else:
                raise ValueError(
                    f"Unsupported history_scope: {profile.get('history_scope')}"
                )

            prepared_sessions = []
            system: TrajectorySystem | None = None
            prediction_text = ""
            trace_rows: list[dict[str, Any]] = []
            accessed_units: list[str] = []
            direct_accessed_units: list[str] = []
            diagnostics: Mapping[str, Any] = {}
            execution_error: str | None = None
            started = time.perf_counter()
            run_namespace = namespace(
                str(run_manifest["run_id"]),
                profile_name,
                trajectory.key,
                question.index,
            )
            info = TrajectoryInfo(
                run_id=str(run_manifest["run_id"]),
                trajectory_id=hashlib.sha256(
                    trajectory.key.encode("utf-8")
                ).hexdigest()[:16],
                namespace=run_namespace,
                persons=trajectory.persons,
                target_person=trajectory.persons[0],
                question_index=question.index,
            )
            staging: tempfile.TemporaryDirectory[str] | None = None

            try:
                source_sessions = visible_history(tuple(source_sessions), question)
                if profile["input_mode"] == "native":
                    staging_root.mkdir(parents=True, exist_ok=True)
                    staging = tempfile.TemporaryDirectory(
                        prefix=f"{run_namespace}_",
                        dir=staging_root,
                    )
                builder = InputBuilder(
                    mode=str(profile.get("input_mode", "caption")),
                    caption_store=shared_caption_store,
                    caption_error=str(profile.get("caption_error", "fail")),
                    media_staging_directory=staging.name if staging else None,
                    media_quality=config.get('media_processing', {}).get('quality', {}),
                )
                prepared_sessions = [fragment for session in source_sessions
                                     for fragment in builder.fragment_session(builder.prepare_session(session))]
                prepared_question = builder.prepare_question(question)
                system = create_system(info, adapter_config)
                runtime_info = system.runtime_info()
                validate_runtime_info(runtime_info, expected_runtime)
                if runtime_info.get("state_namespace") != run_namespace:
                    raise ValueError(
                        "Adapter did not use the namespace supplied by the runner."
                    )
                if runtime_info.get("state_was_empty_before_ingest") is not True:
                    raise ValueError(
                        "Adapter namespace was not empty before trajectory ingestion."
                    )
                for session in prepared_sessions:
                    system.ingest(session)
                prediction = normalize_prediction(system.answer(prepared_question))
                prediction_text = prediction.answer.strip()
                diagnostics = prediction.diagnostics
                if builder.video_split_audit:
                    diagnostics = {**diagnostics, 'video_pre_ingestion': builder.video_split_audit,
                                   'ingested_fragments': len(prepared_sessions)}
                if not prediction_text:
                    raise ValueError("System returned an empty answer.")
                direct_accessed_units = list(prepared_question.source_units)
                valid_units = {
                    unit
                    for session in prepared_sessions
                    for unit in session.source_units
                }
                valid_units.update(prepared_question.source_units)
                trace_rows, traced_units = validate_trace(
                    prediction.retrieval_trace,
                    valid_units,
                    top_k,
                )
                accessed_units = list(
                    dict.fromkeys([*direct_accessed_units, *traced_units])
                )
            except Exception as error:
                execution_error = safe_error(error, root)
            finally:
                if system is not None:
                    try:
                        cleanup = system.close()
                        if not isinstance(cleanup, Mapping):
                            raise TypeError(
                                "close() must return structured cleanup diagnostics."
                            )
                        if cleanup.get("state_namespace") != run_namespace:
                            raise ValueError(
                                "Cleanup diagnostics report a different namespace."
                            )
                        if cleanup.get("state_deleted") is not True:
                            raise ValueError(
                                "Adapter did not confirm deletion of isolated state."
                            )
                    except Exception as error:
                        close_error = safe_error(error, root)
                        execution_error = (
                            f"{execution_error}; cleanup: {close_error}"
                            if execution_error
                            else f"State cleanup failed: {close_error}"
                        )
                if staging is not None:
                    staging.cleanup()

            if execution_error:
                prediction_text = ""
                trace_rows = []
                accessed_units = []
                direct_accessed_units = []
            record = {
                "question_key": key,
                "method": profile_name,
                "family": profile["family"],
                "dimension": trajectory.dimension,
                "task": trajectory.task,
                "record_id": trajectory.record_id,
                "question_index": question.index,
                "question": question.text,
                "question_timestamp": question.timestamp,
                "visible_session_indices": [session.index for session in source_sessions],
                "visible_sessions": len(source_sessions),
                "reference_answer": question.reference_answer,
                "prediction": prediction_text,
                "retrieval_top_k": top_k,
                "retrieval_trace": trace_rows,
                "direct_accessed_units": direct_accessed_units,
                "accessed_units": accessed_units,
                "latency_seconds": round(time.perf_counter() - started, 6),
                "error": execution_error,
                "run_fingerprint": run_manifest["run_fingerprint"],
                "namespace": run_namespace,
            }
            if config["protocol"].get("persist_adapter_diagnostics", False):
                record["diagnostics"] = diagnostics
            writer.append(record)
            existing[key] = record
            processed += 1
        if limit_questions and processed >= limit_questions:
            break

    expected = sum(len(item.questions) for item in trajectories)
    expected_keys = {question_key(t, q.index) for t in trajectories for q in t.questions
                     if question_keys is None or question_key(t, q.index) in question_keys}
    expected = len(expected_keys)
    latest = latest_by_question(output_path)
    if set(latest) - expected_keys:
        raise ValueError('Prediction file contains questions outside this run selection.')
    if staging_root.exists():
        try:
            staging_root.rmdir()
        except OSError:
            pass
    completed = len(latest)
    errors = sum(bool(item.get("error")) for item in latest.values())
    return {
        "profile": profile_name,
        "run_id": run_manifest["run_id"],
        "expected_questions": expected,
        "completed_questions": completed,
        "missing_questions": expected - completed,
        "coverage": completed / expected if expected else 0,
        "is_complete": completed == expected,
        "inference_errors": errors,
        "processed_this_run": processed,
    }
