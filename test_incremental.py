"""Test incremental processing by adding new data"""

import polars as pl
from datetime import datetime, timedelta
from pathlib import Path

# Add new events to the log file
log_path = Path("data/logs/events.parquet")

# Read existing data
existing = pl.read_parquet(log_path)
max_ts = existing['timestamp'].max()

print(f"Existing data: {len(existing)} rows")
print(f"Last timestamp: {max_ts}")

# Create new data with timestamps AFTER the existing data
new_data = pl.DataFrame({
    "user_id": [1, 2, 3, 4] * 5,
    "event": ["login", "click", "purchase", "logout"] * 5,
    "timestamp": [
        max_ts + timedelta(hours=i+1)  # Start 1 hour after last record
        for i in range(20)
    ],
    "value": list(range(200, 220))
})

print(f"\nNew data to add: {len(new_data)} rows")
print(f"New data timestamp range: {new_data['timestamp'].min()} to {new_data['timestamp'].max()}")

# Append new data
combined = pl.concat([existing, new_data])
combined.write_parquet(log_path)

print(f"\nTotal data now: {len(combined)} rows")

# Now run the pipeline again
print("\n" + "=" * 50)
print("Running incremental update...")
print("=" * 50)

from example import app
app.run()

# Check results
print("\n" + "=" * 50)
print("Checking incremental results...")
print("=" * 50)

user_events = pl.read_parquet("data/output/user_events.parquet")
print(f"\nuser_events table: {len(user_events)} rows")
print(f"Expected: {len(combined)} rows")
print(f"Match: {'✓' if len(user_events) == len(combined) else '✗'}")

if len(user_events) == len(combined):
    print("\n✓ Incremental processing works correctly!")
else:
    print(f"\n✗ Mismatch: got {len(user_events)}, expected {len(combined)}")
    print(f"   Difference: {len(combined) - len(user_events)} rows")

# Check state was updated
import json
state_file = Path("data/.pbt/state.json")
with open(state_file) as f:
    state = json.load(f)

print(f"\nState last_max_value: {state['user_events']['last_max_value']}")
expected_max = str(combined['timestamp'].max())
print(f"Expected max value: {expected_max}")
print(f"State correct: {'✓' if state['user_events']['last_max_value'] == expected_max else '✗'}")
