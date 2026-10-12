from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .manifest import hash_json, model_revisions


REQUIRED_MODEL_ROLES = {
    "answer",
    "memory_extraction_and_evolution",
    "caption",
    "judge",
    "text_embedding",
    "multimodal_embedding",
}


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("protocol_version", "dataset", "protocol", "models", "method_profiles"):
        if key not in config:
            raise ValueError(f"Configuration is missing {key}.")
    missing_roles = REQUIRED_MODEL_ROLES - set(config["models"])
    if missing_roles:
        raise ValueError(
            f"Configuration is missing model roles: {sorted(missing_roles)}"
        )
    if int(config["dataset"].get("expected_questions", 0)) <= 0:
        raise ValueError("Dataset configuration must specify a positive question count.")
    if (
        config["protocol"].get("state_isolation")
        != "one fresh system instance per question"
    ):
        raise ValueError("Formal protocol requires one fresh system per question.")


def resolved_profile(
    config: Mapping[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    validate_config(config)
    if profile_name not in config["method_profiles"]:
        raise KeyError(f"Unknown method profile: {profile_name}")
    profile = deepcopy(config["method_profiles"][profile_name])
    defaults = config.get("retrieval_defaults", {})
    if profile.get("family") in {"retrieval", "explicit_memory"}:
        profile.setdefault("retrieval_unit", defaults.get("unit", "session"))
        profile.setdefault("retrieval_top_k", defaults.get("final_top_k", 10))
        profile.setdefault("dense_candidates", defaults.get("dense_candidates", 10))
        profile.setdefault("lexical_candidates", defaults.get("lexical_candidates", 10))
        profile.setdefault("rrf_k", defaults.get("rrf_k", 60))
    return profile


def expected_runtime_info(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    framework_version: str,
) -> dict[str, Any]:
    answer = config["models"]["answer"]
    revisions = model_revisions(config)
    expected = {
        "framework_version": framework_version,
        "answer_model": answer["model"],
        "answer_model_revision": revisions["answer"],
        "answer_chat_template_revision": answer["chat_template_revision"],
        "answer_configuration_fingerprint": hash_json(answer),
        "resolved_profile_fingerprint": hash_json(profile),
        "input_mode": profile["input_mode"],
        "retrieval_top_k": profile.get("retrieval_top_k"),
        "session_chunk_size": profile.get("session_chunk_size", 1),
        "thinking_enabled": bool(answer.get("enable_thinking")),
        "random_seed": int(config["protocol"]["random_seed"]),
        "answer_prompt_fingerprint": hash_json(config.get("prompts", {}).get("answer")),
        "history_serialization_fingerprint": hash_json(
            {
                "format": config.get("prompts", {}).get("history_serialization"),
                "roles": config.get("prompts", {}).get("answer_message_roles"),
            }
        ),
    }
    family = profile["family"]
    if family in {"retrieval", "explicit_memory"}:
        expected["retrieval_unit"] = profile["retrieval_unit"]
        expected["retrieval_defaults_fingerprint"] = hash_json(
            config.get("retrieval_defaults", {})
        )
    if family == "retrieval":
        embedding_roles = {
            role
            for role in (
                profile.get("embedding_role"),
                profile.get("text_embedding_role"),
                profile.get("visual_embedding_role"),
            )
            if role
        }
        expected["embedding_configs"] = {
            role: {
                **dict(config["models"][role]),
                "revision": revisions[role],
            }
            for role in sorted(embedding_roles)
        }
    if family == "explicit_memory":
        memory_role = profile["memory_model_role"]
        memory_model = config["models"][memory_role]
        expected.update(
            {
                "memory_model": memory_model["model"],
                "memory_model_revision": revisions[memory_role],
                "memory_write_unit": profile["memory_write_unit"],
                "memory_owner": profile["memory_owner"],
            }
        )
    if profile["input_mode"] == "native":
        expected["media_processing_fingerprint"] = hash_json(
            config.get("media_processing", {})
        )
    return expected


def validate_runtime_info(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    missing = set(expected) - set(actual)
    if missing:
        raise ValueError(f"Adapter runtime_info is missing: {sorted(missing)}")
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"Adapter runtime configuration mismatch: {mismatches}")
