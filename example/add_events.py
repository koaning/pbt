"""
Add new events to the raw.csv file.

This script appends new events to the raw event stream for testing
incremental processing.
"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta


def add_events(events: list[dict], raw_file: str = "raw.csv") -> None:
    """Append new events to the raw CSV file."""
    script_dir = Path(__file__).parent
    raw_path = script_dir / raw_file

    # Convert to DataFrame
    new_df = pl.DataFrame(events)

    # Append to existing file
    with open(raw_path, 'a') as f:
        new_df.write_csv(f, include_header=False)

    print(f"Added {len(events)} new events to {raw_file}")


def generate_sample_events(user: str, base_time: datetime, session_minutes: int = 10) -> list[dict]:
    """Generate a realistic session of events for a user."""
    events = []
    current_time = base_time

    # Login
    events.append({
        'user': user,
        'event': 'login',
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Browse
    for i in range(2):
        current_time += timedelta(minutes=1, seconds=30)
        events.append({
            'user': user,
            'event': 'page_view',
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
        })

    # Click
    current_time += timedelta(seconds=45)
    events.append({
        'user': user,
        'event': 'click_button',
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Logout
    current_time += timedelta(minutes=session_minutes - 5)
    events.append({
        'user': user,
        'event': 'logout',
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
    })

    return events


if __name__ == "__main__":
    # Example: Add new events for existing users
    new_events = []

    # Alice comes back in the evening
    new_events.extend(generate_sample_events('alice', datetime(2025, 1, 15, 18, 30, 0)))

    # New user Eve joins
    new_events.extend(generate_sample_events('eve', datetime(2025, 1, 15, 19, 0, 0)))

    # Bob makes a quick visit
    new_events.extend(generate_sample_events('bob', datetime(2025, 1, 15, 20, 15, 0), session_minutes=5))

    add_events(new_events)
    print("\nNew events added:")
    print(pl.DataFrame(new_events))
