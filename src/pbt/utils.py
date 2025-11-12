"""Utility helpers for PBT internals."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_timestamp_value(value: Any):
    """Best-effort parsing of ISO-like timestamp strings for Polars literals."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace(" ", "T"))
        except ValueError:
            return value

    return value


def coerce_datetime(value: Any, label: str = "value") -> datetime:
    """Ensure rerun boundaries are valid datetimes."""
    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace(" ", "T"))
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO-8601 datetime string") from exc

    raise TypeError(f"{label} must be a datetime or ISO-8601 string")
