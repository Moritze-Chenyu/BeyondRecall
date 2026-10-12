from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Iterator

from .models import MediaAsset, Question, Session, Trajectory, Turn


DIMENSIONS = ("retrieval", "reasoning", "management")


def opaque_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "document"
    if suffix in {".xlsx", ".xlsm", ".xls", ".ods"}:
        return "spreadsheet"
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        return "video"
    mime = mimetypes.guess_type(path.name)[0] or ""
    return "image" if mime.startswith("image/") else "file"


class BeyondRecallDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

    def task_files(self) -> list[Path]:
        files = [
            path
            for dimension in DIMENSIONS
            for path in sorted((self.root / dimension).glob("*.json"))
        ]
        if len(files) != 11:
            raise ValueError(f"Expected 11 task files, found {len(files)}")
        return files

    def _asset(self, value: str, task: str) -> MediaAsset:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Media path must be relative: {value}")
        if len(relative.parts) != 3 or relative.parts[1] != task:
            raise ValueError(f"Invalid task-local media path: {value}")
        local = (self.root / relative).resolve()
        if not local.is_relative_to(self.root):
            raise ValueError("Media symlink escapes dataset root.")
        if not local.is_file():
            raise FileNotFoundError(f"Missing media file: {value}")
        return MediaAsset(
            asset_id=opaque_id("media", value),
            relative_path=value,
            local_path=local,
            media_type=media_type(local),
        )

    def trajectories(self) -> Iterator[Trajectory]:
        for task_file in self.task_files():
            dimension = task_file.parent.name
            task = task_file.stem
            records = json.loads(task_file.read_text(encoding="utf-8"))
            if [record.get("id") for record in records] != list(range(len(records))):
                raise ValueError(f"Record IDs are not contiguous in {task_file.name}")

            for record in records:
                record_id = int(record["id"])
                trajectory_key = f"{dimension}/{task}/{record_id}"
                persons = tuple(record["persons"])
                if len(persons) != 2:
                    raise ValueError(
                        f"Expected two primary persons in {task_file.name}"
                    )
                selector_to_unit: dict[str, str] = {}
                sessions = []
                for session_index, raw_session in enumerate(record["conversation"]):
                    session_id = opaque_id(
                        "session",
                        f"{trajectory_key}/session/{session_index}",
                    )
                    selector_to_unit[f"session:{session_index}"] = session_id
                    turns = []
                    for turn_index, raw_turn in enumerate(raw_session["turns"]):
                        turn_id = opaque_id(
                            "turn",
                            f"{trajectory_key}/session/{session_index}/turn/{turn_index}",
                        )
                        selector_to_unit[f"turn:{session_index}:{turn_index}"] = turn_id
                        assets = tuple(
                            self._asset(value, task)
                            for value in raw_turn.get("media", [])
                        )
                        for asset in assets:
                            selector_to_unit[asset.relative_path] = asset.asset_id
                        turns.append(
                            Turn(
                                index=turn_index,
                                unit_id=turn_id,
                                person=str(raw_turn["person"]),
                                content=str(raw_turn["content"]),
                                media=assets,
                                timestamp=raw_turn.get("timestamp"),
                            )
                        )
                    sessions.append(
                        Session(
                            index=session_index,
                            unit_id=session_id,
                            date=str(raw_session["date"]),
                            turns=tuple(turns),
                        )
                    )

                questions = []
                for question_index, raw_question in enumerate(record["questions"]):
                    assets = tuple(
                        self._asset(value, task)
                        for value in raw_question.get("media", [])
                    )
                    for asset in assets:
                        selector_to_unit[asset.relative_path] = asset.asset_id
                    questions.append(
                        Question(
                            index=question_index,
                            text=str(raw_question["question"]),
                            reference_answer=str(raw_question["answer"]),
                            media=assets,
                            timestamp=raw_question.get("timestamp"),
                        )
                    )

                yield Trajectory(
                    dimension=dimension,
                    task=task,
                    record_id=record_id,
                    persons=(str(persons[0]), str(persons[1])),
                    sessions=tuple(sessions),
                    questions=tuple(questions),
                    selector_to_unit=selector_to_unit,
                )

    def expected_questions(self) -> list[dict[str, object]]:
        return [
            {
                "question_key": f"{item.key}/q{question.index}",
                "dimension": item.dimension,
                "task": item.task,
                "record_id": item.record_id,
                "question_index": question.index,
            }
            for item in self.trajectories()
            for question in item.questions
        ]

    def count_questions(self) -> int:
        return len(self.expected_questions())
