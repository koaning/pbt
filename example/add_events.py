"""
Add synthetic events to the raw.csv file.

This script appends new events to the raw event stream for testing
incremental processing. By default it always targets a calendar day that is
newer than the dates already present in ``raw.csv`` so each run grows the
dataset chronologically. Pass ``--date YYYY-MM-DD`` to override that behavior
and insert events for a specific day.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

RAW_FILE = "raw.csv"


def add_events(events: list[dict], raw_file: str = RAW_FILE) -> None:
    """Append new events to the raw CSV file."""
    script_dir = Path(__file__).parent
    raw_path = script_dir / raw_file

    # Convert to DataFrame
    new_df = pl.DataFrame(events)

    # Append to existing file
    with open(raw_path, "a") as f:
        new_df.write_csv(f, include_header=False)

    print(f"Added {len(events)} new events to {raw_file}")


def generate_sample_events(
    user: str, base_time: datetime, session_minutes: int = 10
) -> list[dict]:
    """Generate a realistic session of events for a user."""
    events = []
    current_time = base_time

    # Login
    events.append(
        {
            "user": user,
            "event": "login",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Browse
    for i in range(2):
        current_time += timedelta(minutes=1, seconds=30)
        events.append(
            {
                "user": user,
                "event": "page_view",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    # Click
    current_time += timedelta(seconds=45)
    events.append(
        {
            "user": user,
            "event": "click_button",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Logout
    current_time += timedelta(minutes=session_minutes - 5)
    events.append(
        {
            "user": user,
            "event": "logout",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return events


def get_latest_event_date(raw_path: Path) -> date | None:
    """Return the latest event date stored in the raw CSV, if available."""
    if not raw_path.exists():
        return None

    df = pl.read_csv(raw_path, columns=["timestamp"])
    if df.height == 0:
        return None

    max_timestamp = df.select(pl.col("timestamp").max()).item()
    if max_timestamp is None:
        return None

    return datetime.strptime(max_timestamp, "%Y-%m-%d %H:%M:%S").date()


def determine_target_date(raw_path: Path, forced_date: str | None) -> date:
    """Decide which day we should generate events for."""
    latest_existing = get_latest_event_date(raw_path)

    if forced_date:
        target = datetime.strptime(forced_date, "%Y-%m-%d").date()
        if latest_existing and target <= latest_existing:
            print(
                "Warning: provided date is not after the latest existing date "
                f"({target} <= {latest_existing})."
            )
        return target

    if latest_existing:
        return latest_existing + timedelta(days=1)

    # No data yet, so start from today.
    return date.today()


def build_new_events_for_date(target_date: date) -> list[dict]:
    """Generate a batch of demo events that live within target_date."""

    def day_time(hour: int, minute: int = 0, second: int = 0) -> datetime:
        return datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            hour,
            minute,
            second,
        )

    new_events: list[dict] = []
    new_events.extend(generate_sample_events("alice", day_time(18, 30, 0)))
    new_events.extend(generate_sample_events("eve", day_time(19, 0, 0)))
    new_events.extend(
        generate_sample_events("bob", day_time(20, 15, 0), session_minutes=5)
    )
    return new_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append synthetic events to raw.csv")
    parser.add_argument(
        "--date",
        help="Date (YYYY-MM-DD) to assign to the generated events. Defaults to the day after the latest entry.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).parent
    raw_path = script_dir / RAW_FILE

    target_date = determine_target_date(raw_path, args.date)

    new_events = build_new_events_for_date(target_date)
    add_events(new_events)

    print(f"\nGenerated events for {target_date}. Summary:")
    print(
        pl.read_csv(raw_path)
        .group_by(pl.col("timestamp").str.slice(0, 10))
        .len()
        .sort("timestamp")
    )
