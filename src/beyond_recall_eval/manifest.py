from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def hash_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(dataset_root: str | Path) -> str:
    root = Path(dataset_root)
    digest = hashlib.sha256()
    for dimension in ("retrieval", "reasoning", "management"):
        for path in sorted((root / dimension).glob("*.json")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(hash_file(path) or ""))
    checksums = root / "SHA256SUMS"
    if checksums.is_file():
        digest.update(bytes.fromhex(hash_file(checksums) or ""))
    return digest.hexdigest()


def code_fingerprint(package_root: str | Path) -> str:
    root = Path(package_root)
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(hash_file(path) or ""))
    return digest.hexdigest()


def model_revisions(config: Mapping[str, Any]) -> dict[str, str]:
    revisions = {}
    for role, value in config["models"].items():
        if not isinstance(value, Mapping) or "model" not in value:
            continue
        revision = str(value.get("revision", "")).strip()
        revision_env = str(value.get("revision_env", "")).strip()
        if revision_env:
            revision = os.getenv(revision_env, revision).strip()
        if not revision:
            raise ValueError(f"Model revision is not fixed for role {role}.")
        revisions[role] = revision
    return revisions


def deployment_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    fingerprints = {}
    for role, value in config["models"].items():
        if not isinstance(value, Mapping) or "model" not in value:
            continue
        base_url_env = str(value.get("base_url_env", "")).strip()
        endpoint = os.getenv(base_url_env, "").strip() if base_url_env else ""
        fingerprints[role] = (
            hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            if endpoint
            else "local-or-unset"
        )
    return fingerprints


def build_inference_manifest(
    *,
    package_root: str | Path,
    dataset_root: str | Path,
    config: Mapping[str, Any],
    profile_name: str,
    resolved_profile: Mapping[str, Any],
    adapter_file: str | Path,
    framework_lock: str | Path,
    caption_store: str | Path | None,
    evidence_manifest: str | Path | None,
    framework_version: str,
    run_id: str | None,
    sample_manifest: str | Path | None = None,
) -> dict[str, Any]:
    if not framework_version.strip():
        raise ValueError("A non-empty framework version or commit is required.")
    manifest = {
        "manifest_version": 2,
        "run_id": run_id or uuid.uuid4().hex,
        "profile": profile_name,
        "framework_version": framework_version,
        "sample_manifest_sha256": hash_file(sample_manifest),
        "dataset_fingerprint": dataset_fingerprint(dataset_root),
        "evaluation_code_fingerprint": code_fingerprint(package_root),
        "configuration_fingerprint": hash_json(config),
        "resolved_profile": resolved_profile,
        "adapter": {
            "filename": Path(adapter_file).name,
            "sha256": hash_file(adapter_file),
        },
        "framework_dependency_lock": {
            "filename": Path(framework_lock).name,
            "sha256": hash_file(framework_lock),
        },
        "caption_cache": {
            "filename": Path(caption_store).name if caption_store else None,
            "sha256": hash_file(caption_store),
        },
        "evidence_manifest": {
            "filename": Path(evidence_manifest).name if evidence_manifest else None,
            "sha256": hash_file(evidence_manifest),
        },
        "model_revisions": model_revisions(config),
        "deployment_fingerprints": deployment_fingerprints(config),
    }
    video_manifest = config.get('media_processing', {}).get('quality', {}).get('video_segment_manifest')
    if video_manifest and resolved_profile.get('input_mode') == 'native':
        digest = hash_file(video_manifest)
        if digest != config['media_processing']['quality'].get('video_segment_manifest_sha256'):
            raise ValueError('Video segment manifest is not pinned to its current SHA.')
        manifest['video_segment_cache'] = {'filename': Path(video_manifest).name, 'sha256': digest}
    manifest["run_fingerprint"] = hash_json(manifest)
    return manifest


def write_or_validate_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                "Resume refused because the effective run manifest changed."
            )
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return dict(manifest)
