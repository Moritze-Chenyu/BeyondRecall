from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from .configuration import resolved_profile, validate_config
from .dataset import BeyondRecallDataset
from .evidence import EvidenceManifest
from .judging import run_judging
from .manifest import build_inference_manifest, write_or_validate_manifest, dataset_fingerprint, hash_file, hash_json
from .metrics import write_summary
from .rendering import validate_caption_store
from .runner import run_inference
from .sampling import select_trajectories


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_create_system(adapter_file: Path, function_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "beyond_recall_user_adapter",
        adapter_file,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter: {adapter_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise AttributeError(f"Adapter must define callable {function_name}().")
    return function


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_common_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--allow-partial-evidence", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beyond Recall evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="Run framework inference only.")
    add_common_dataset_arguments(infer)
    infer.add_argument("--adapter-file", type=Path, required=True)
    infer.add_argument("--adapter-function", default="create_system")
    infer.add_argument("--framework-version", required=True)
    infer.add_argument("--framework-lock", type=Path, required=True)
    infer.add_argument("--caption-store", type=Path)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--run-manifest", type=Path)
    infer.add_argument("--run-id")
    infer.add_argument("--limit-questions", type=int, default=0)
    infer.add_argument("--retry-errors", action="store_true")

    judge = commands.add_parser("judge", help="Judge saved predictions only.")
    judge.add_argument("--config", type=Path, required=True)
    judge.add_argument("--predictions", type=Path, required=True)
    judge.add_argument("--inference-manifest", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--judge-manifest", type=Path)
    judge.add_argument("--judge-mode", choices=["llm", "exact", "none"], default="llm")
    judge.add_argument("--limit-questions", type=int, default=0)
    judge.add_argument("--retry-errors", action="store_true")

    summary = commands.add_parser("summarize", help="Build strict metrics.")
    add_common_dataset_arguments(summary)
    summary.add_argument("--predictions", type=Path, required=True)
    summary.add_argument("--judgments", type=Path)
    summary.add_argument("--inference-manifest", type=Path, required=True)
    summary.add_argument("--judge-manifest", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evidence_for_args(
    args: argparse.Namespace,
    trajectories: list[Any],
    profile: dict[str, Any],
) -> EvidenceManifest | None:
    is_oracle = profile.get("history_scope") == "private_annotated_evidence_only"
    if is_oracle and not args.evidence_manifest:
        raise ValueError("Oracle profiles require --evidence-manifest.")
    if not args.evidence_manifest:
        return None
    return EvidenceManifest(
        args.evidence_manifest,
        trajectories,
        require_complete=is_oracle or not args.allow_partial_evidence,
    )


def infer_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    profile = resolved_profile(config, args.profile)
    if args.adapter_file.resolve() in {
        (PACKAGE_ROOT / 'adapters/builtin.py').resolve(),
        (PACKAGE_ROOT / 'src/beyond_recall_eval/systems.py').resolve(),
    }:
        from .systems import require_builtin_profile
        require_builtin_profile(profile)
    if profile["input_mode"] == "caption" and not args.caption_store:
        raise ValueError("Caption profiles require --caption-store.")
    if profile["input_mode"] == "caption":
        validate_caption_store(args.caption_store, config)
    dataset = BeyondRecallDataset(args.dataset)
    trajectories = list(dataset.trajectories())
    trajectories = select_trajectories(trajectories, args.sample_manifest, args.dataset)
    evidence = evidence_for_args(args, trajectories, profile)
    create_system = load_create_system(
        args.adapter_file.resolve(),
        args.adapter_function,
    )
    manifest_path = args.run_manifest or args.output.with_suffix(".run.json")
    if args.output.exists() and not manifest_path.exists():
        raise ValueError(
            "Predictions exist without their run manifest; resume is refused."
        )
    run_id = args.run_id
    if manifest_path.exists() and not run_id:
        run_id = load_manifest(manifest_path)["run_id"]
    manifest = build_inference_manifest(
        package_root=PACKAGE_ROOT,
        dataset_root=args.dataset,
        config=config,
        profile_name=args.profile,
        resolved_profile=profile,
        adapter_file=args.adapter_file,
        framework_lock=args.framework_lock,
        caption_store=args.caption_store,
        evidence_manifest=args.evidence_manifest,
        framework_version=args.framework_version,
        run_id=run_id,
        sample_manifest=args.sample_manifest,
    )
    write_or_validate_manifest(manifest_path, manifest)
    result = run_inference(
        dataset_root=args.dataset,
        trajectories=trajectories,
        config=config,
        profile_name=args.profile,
        profile=profile,
        create_system=create_system,
        output_path=args.output,
        run_manifest=manifest,
        framework_version=args.framework_version,
        caption_store_path=args.caption_store,
        evidence_manifest=evidence,
        limit_questions=args.limit_questions,
        retry_errors=args.retry_errors,
    )
    coverage_path = args.output.with_suffix(".coverage.json")
    coverage_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def judge_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    inference_manifest = load_manifest(args.inference_manifest)
    judge_manifest = args.judge_manifest or args.output.with_suffix(".run.json")
    return run_judging(
        predictions_path=args.predictions,
        judgments_path=args.output,
        judge_manifest_path=judge_manifest,
        inference_manifest=inference_manifest,
        config=config,
        judge_mode=args.judge_mode,
        retry_errors=args.retry_errors,
        limit_questions=args.limit_questions,
    )


def summary_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    profile = resolved_profile(config, args.profile)
    trajectories = list(BeyondRecallDataset(args.dataset).trajectories())
    trajectories = select_trajectories(trajectories, args.sample_manifest, args.dataset)
    evidence = evidence_for_args(args, trajectories, profile)
    inference_manifest = load_manifest(args.inference_manifest)
    judge_manifest = load_manifest(args.judge_manifest) if args.judge_manifest else None
    if inference_manifest['dataset_fingerprint'] != dataset_fingerprint(args.dataset):
        raise ValueError('Summary dataset does not match the inference release.')
    if inference_manifest.get('sample_manifest_sha256') != hash_file(args.sample_manifest):
        raise ValueError('Summary sample manifest does not match the inference selection.')
    if inference_manifest['configuration_fingerprint'] != hash_json(config) or inference_manifest['profile'] != args.profile:
        raise ValueError('Summary configuration/profile does not match inference.')
    return write_summary(
        args.output,
        trajectories=trajectories,
        predictions_path=args.predictions,
        judgments_path=args.judgments,
        inference_manifest=inference_manifest,
        judge_manifest=judge_manifest,
        evidence_manifest=evidence,
        profile=profile,
    )


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        result = infer_command(args)
    elif args.command == "judge":
        result = judge_command(args)
    else:
        result = summary_command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
