from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .models import (
    PreparedQuestion,
    PreparedSession,
    RetrievalTraceItem,
    SystemPrediction,
)


@runtime_checkable
class TrajectorySystem(Protocol):
    """One isolated, single-question system instance."""

    def runtime_info(self) -> Mapping[str, Any]:
        """Report the effective runtime configuration for protocol validation."""

    def ingest(self, session: PreparedSession) -> None:
        """Ingest a session or atomic video/transcript fragment in chronological order."""

    def answer(self, question: PreparedQuestion) -> SystemPrediction:
        """Answer without access to the reference answer or hidden annotations."""

    def close(self) -> Mapping[str, Any]:
        """Delete state and confirm it was removed from the supplied namespace."""


def normalize_trace_item(value: Any) -> RetrievalTraceItem:
    if isinstance(value, RetrievalTraceItem):
        return value
    if isinstance(value, dict):
        return RetrievalTraceItem(
            rank=int(value["rank"]),
            source_units=tuple(str(item) for item in value.get("source_units", [])),
            delivered_to_answer=bool(value.get("delivered_to_answer", True)),
            score=(float(value["score"]) if value.get("score") is not None else None),
        )
    raise TypeError("retrieval_trace items must be RetrievalTraceItem or dict")


def normalize_prediction(value: Any) -> SystemPrediction:
    if isinstance(value, SystemPrediction):
        return value
    if isinstance(value, str):
        return SystemPrediction(answer=value)
    if isinstance(value, dict):
        trace = value.get("retrieval_trace", [])
        return SystemPrediction(
            answer=str(value.get("answer", "")),
            retrieval_trace=tuple(normalize_trace_item(item) for item in trace),
            diagnostics=dict(value.get("diagnostics", {})),
        )
    raise TypeError("answer() must return SystemPrediction, str, or a compatible dict")


def trace_item(
    rank: int,
    session: PreparedSession,
    *,
    score: float | None = None,
    delivered_to_answer: bool = True,
) -> RetrievalTraceItem:
    """Create an auditable trace item for one returned session."""

    return RetrievalTraceItem(
        rank=rank,
        source_units=session.source_units,
        delivered_to_answer=delivered_to_answer,
        score=score,
    )
