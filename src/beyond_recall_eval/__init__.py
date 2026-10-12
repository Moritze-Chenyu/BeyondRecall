"""Framework-agnostic evaluation utilities for Beyond Recall."""

from .interface import TrajectorySystem, trace_item
from .models import (
    PreparedQuestion,
    PreparedSession,
    RetrievalTraceItem,
    SystemPrediction,
    TrajectoryInfo,
)

__all__ = [
    "PreparedQuestion",
    "PreparedSession",
    "RetrievalTraceItem",
    "SystemPrediction",
    "TrajectoryInfo",
    "TrajectorySystem",
    "trace_item",
]
