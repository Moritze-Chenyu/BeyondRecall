from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Session, Trajectory
from .temporal import visible_history


@dataclass(frozen=True)
class EvidenceEntry:
    question_key: str
    selectors: tuple[str, ...]
    unit_ids: frozenset[str]


class EvidenceManifest:
    def __init__(
        self,
        path: str | Path,
        trajectories: Iterable[Trajectory],
        *,
        require_complete: bool,
    ):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw_items = payload.get("items", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_items, dict):
            raise ValueError("Evidence manifest must contain an object named items.")

        self.entries: dict[str, EvidenceEntry] = {}
        expected_keys = set()
        for trajectory in trajectories:
            for question in trajectory.questions:
                key = f"{trajectory.key}/q{question.index}"
                expected_keys.add(key)
                raw = raw_items.get(key)
                if raw is None:
                    continue
                selectors = (
                    raw.get("required_units", raw) if isinstance(raw, dict) else raw
                )
                if not isinstance(selectors, list) or not selectors:
                    raise ValueError(f"Evidence entry must be a non-empty list: {key}")
                normalized = tuple(str(item) for item in selectors)
                unknown = [
                    item
                    for item in normalized
                    if item not in trajectory.selector_to_unit
                ]
                if unknown:
                    raise ValueError(f"Unknown evidence selectors for {key}: {unknown}")
                allowed = {u for s in visible_history(trajectory.sessions, question) for u in s.source_units()}
                allowed.update(a.asset_id for a in question.media)
                if {trajectory.selector_to_unit[item] for item in normalized} - allowed:
                    raise ValueError(f"Evidence for {key} refers to history after its query cutoff.")
                self.entries[key] = EvidenceEntry(
                    question_key=key,
                    selectors=normalized,
                    unit_ids=frozenset(
                        trajectory.selector_to_unit[item] for item in normalized
                    ),
                )

        extra = set(raw_items) - expected_keys
        missing = expected_keys - set(self.entries)
        if extra:
            raise ValueError(
                f"Evidence manifest has unknown question keys: {len(extra)}"
            )
        if require_complete and missing:
            raise ValueError(
                f"Evidence manifest is incomplete: {len(missing)} questions missing."
            )
        self.expected_questions = len(expected_keys)
        self.annotated_questions = len(self.entries)

    def entry(self, question_key: str) -> EvidenceEntry | None:
        return self.entries.get(question_key)

    def oracle_sessions(
        self,
        trajectory: Trajectory,
        question_index: int,
    ) -> tuple[Session, ...]:
        key = f"{trajectory.key}/q{question_index}"
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError(f"Oracle evidence is missing for {key}")

        selected_sessions: set[int] = set()
        selected_turns: set[tuple[int, int]] = set()
        selected_media = {
            selector
            for selector in entry.selectors
            if selector.startswith(("images/", "videos/"))
        }
        for selector in entry.selectors:
            if selector.startswith("session:"):
                selected_sessions.add(int(selector.split(":")[1]))
            elif selector.startswith("turn:"):
                _, session_index, turn_index = selector.split(":")
                selected_turns.add((int(session_index), int(turn_index)))

        for session in trajectory.sessions:
            for turn in session.turns:
                if any(asset.relative_path in selected_media for asset in turn.media):
                    selected_turns.add((session.index, turn.index))

        output = []
        for session in trajectory.sessions:
            if session.index in selected_sessions:
                output.append(session)
                continue
            turns = tuple(
                turn
                for turn in session.turns
                if (session.index, turn.index) in selected_turns
            )
            if turns:
                output.append(
                    Session(
                        index=session.index,
                        unit_id=session.unit_id,
                        date=session.date,
                        turns=turns,
                    )
                )
        return tuple(output)
