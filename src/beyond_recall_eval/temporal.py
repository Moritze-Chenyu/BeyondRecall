"""Filter history before a framework, captioner, or staging sees it."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from .models import Question, Session


def parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Timestamp must be a nonempty string.")
    value = value.strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M:%S %p on %d %B, %Y",
                    "%H:%M on %d %B, %Y", "%d %B %Y %H:%M"):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError("Unsupported timestamp format; refusing to expose full history.")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def visible_history(sessions: tuple[Session, ...], question: Question) -> tuple[Session, ...]:
    cutoff = parse_timestamp(question.timestamp) if question.timestamp is not None else None
    visible = []
    for session in sessions:
        date = parse_timestamp(session.date)
        turns = []
        for turn in session.turns:
            timestamp = parse_timestamp(turn.timestamp) if turn.timestamp is not None else date
            if cutoff is None or timestamp <= cutoff:
                turns.append((timestamp, turn.index, turn))
        if turns:
            turns.sort(key=lambda entry: (entry[0], entry[1]))
            visible_date = min(entry[0] for entry in turns) if cutoff and date > cutoff else date
            selected = replace(session, date=visible_date.isoformat() if visible_date != date else session.date,
                               turns=tuple(entry[2] for entry in turns))
            visible.append((visible_date, session.index, selected))
    visible.sort(key=lambda entry: (entry[0], entry[1]))
    return tuple(entry[2] for entry in visible)
