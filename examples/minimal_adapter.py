"""Minimal example of the evaluation adapter interface."""

from __future__ import annotations

from beyond_recall_eval import SystemPrediction, trace_item


class MinimalSystem:
    def __init__(self, trajectory, method_config) -> None:
        self.sessions = []
        self.namespace = trajectory.namespace
        self._runtime_info = {
            **method_config["runtime_expectations"],
            "state_namespace": self.namespace,
            "state_was_empty_before_ingest": True,
        }

    def runtime_info(self):
        return self._runtime_info

    def ingest(self, session) -> None:
        self.sessions.append(session)

    def answer(self, question) -> SystemPrediction:
        return SystemPrediction(
            answer="It cannot be determined from this example adapter.",
            retrieval_trace=tuple(
                trace_item(rank, session)
                for rank, session in enumerate(self.sessions, start=1)
            ),
        )

    def close(self):
        self.sessions.clear()
        return {
            "state_namespace": self.namespace,
            "state_deleted": True,
        }


def create_system(trajectory, method_config):
    return MinimalSystem(trajectory, method_config)
