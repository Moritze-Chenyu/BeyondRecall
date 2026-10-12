from __future__ import annotations

import base64
import mimetypes
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MediaAsset:
    """Dataset-internal media record. This type is never passed to an adapter."""

    asset_id: str
    relative_path: str
    local_path: Path
    media_type: str


@dataclass(frozen=True)
class MediaInput:
    """Opaque native-media handle exposed to a trusted method adapter."""

    asset_id: str
    media_type: str
    suffix: str
    _source_path: Path = field(repr=False, compare=False)
    text_in_history: bool = False
    pdf_representation: str | None = None
    pdf_original_pages: tuple[int, ...] = ()
    pdf_original_page_count: int | None = None
    pdf_source_sha256: str | None = None
    video_source_asset_id: str | None = None
    video_source_sha256: str | None = None
    video_segment_index: int | None = None
    video_start_seconds: float | None = None
    video_end_seconds: float | None = None
    video_frame_count: int | None = None
    video_transcript_status: str | None = None

    def read_bytes(self) -> bytes:
        return self._source_path.read_bytes()

    def as_data_url(self) -> str:
        mime = mimetypes.guess_type(f"media{self.suffix}")[0]
        mime = mime or "application/octet-stream"
        encoded = base64.b64encode(self.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def materialize(self, directory: str | Path) -> Path:
        """Create an opaque local filename without exposing dataset task names."""

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / f"{self.asset_id}{self.suffix}"
        if destination.exists():
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            os.link(self._source_path, temporary)
        except OSError:
            shutil.copyfile(self._source_path, temporary)
        os.replace(temporary, destination)
        return destination


@dataclass(frozen=True)
class Turn:
    index: int
    unit_id: str
    person: str
    content: str
    media: tuple[MediaAsset, ...]
    timestamp: str | None = None


@dataclass(frozen=True)
class Session:
    index: int
    unit_id: str
    date: str
    turns: tuple[Turn, ...]

    def source_units(self, include_media: bool = True) -> tuple[str, ...]:
        units = [self.unit_id]
        for turn in self.turns:
            units.append(turn.unit_id)
            if include_media:
                units.extend(asset.asset_id for asset in turn.media)
        return tuple(dict.fromkeys(units))


@dataclass(frozen=True)
class Question:
    index: int
    text: str
    reference_answer: str
    media: tuple[MediaAsset, ...]
    timestamp: str | None = None


@dataclass(frozen=True)
class Trajectory:
    dimension: str
    task: str
    record_id: int
    persons: tuple[str, str]
    sessions: tuple[Session, ...]
    questions: tuple[Question, ...]
    selector_to_unit: dict[str, str]

    @property
    def key(self) -> str:
        return f"{self.dimension}/{self.task}/{self.record_id}"

    @property
    def all_unit_ids(self) -> set[str]:
        values: set[str] = set()
        for session in self.sessions:
            values.update(session.source_units())
        for question in self.questions:
            values.update(asset.asset_id for asset in question.media)
        return values


@dataclass(frozen=True)
class TrajectoryInfo:
    run_id: str
    trajectory_id: str
    namespace: str
    persons: tuple[str, str]
    target_person: str
    question_index: int


@dataclass(frozen=True)
class PreparedTurn:
    index: int
    person: str
    content: str
    media: tuple[MediaInput, ...]
    source_units: tuple[str, ...]
    timestamp: str | None = None


@dataclass(frozen=True)
class PreparedSession:
    index: int
    date: str
    turns: tuple[PreparedTurn, ...]
    text: str
    media: tuple[MediaInput, ...]
    source_units: tuple[str, ...]


@dataclass(frozen=True)
class PreparedQuestion:
    index: int
    text: str
    media: tuple[MediaInput, ...]
    source_units: tuple[str, ...]
    timestamp: str | None = None


@dataclass(frozen=True)
class RetrievalTraceItem:
    rank: int
    source_units: tuple[str, ...]
    delivered_to_answer: bool = True
    score: float | None = None


@dataclass(frozen=True)
class SystemPrediction:
    answer: str
    retrieval_trace: tuple[RetrievalTraceItem, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)
