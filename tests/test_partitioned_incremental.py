"""Test for incremental + partitioned append bug."""
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from pbt import LocalSink, conf


def test_incremental_partitioned_append_no_duplicates():
    """
    Bug: When using incremental_table with partition_by + partition_mode="append",
    data gets duplicated because:
    1. core._apply_incremental_merge() reads existing + concats new
    2. sink._write_partitioned() ALSO appends to existing partition files

    Expected: 5 rows total after two runs
    Actual (with bug): 8 rows (3 initial + 5 on second run due to double-append)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output_dir = root / "output"

        # Create sample data file with timestamps
        data_file = root / "events.parquet"
        initial_data = pl.DataFrame({
            "id": [1, 2, 3],
            "timestamp": [
                datetime(2025, 1, 15, 10, 0),
                datetime(2025, 1, 15, 11, 0),
                datetime(2025, 1, 16, 10, 0),
            ],
            "value": ["a", "b", "c"],
        })
        initial_data.write_parquet(data_file)

        # Setup PBT app
        app = conf(root=root)
        sink = LocalSink(path=output_dir)

        @app.source
        def raw_events():
            return pl.scan_parquet(data_file).with_columns(
                pl.col("timestamp").dt.date().alias("date")
            )

        @app.incremental_table(
            time_column="timestamp",
            partition_by="date",
            partition_mode="append",
            sink=sink,
        )
        def events(raw_events):
            return raw_events

        # First run: should write 3 rows
        app.run(silent=True)

        # Verify initial state
        result1 = sink.read("events").collect()
        assert len(result1) == 3, f"Expected 3 rows after first run, got {len(result1)}"

        # Add more data with newer timestamps
        new_data = pl.DataFrame({
            "id": [4, 5],
            "timestamp": [
                datetime(2025, 1, 16, 12, 0),  # Same date as id=3
                datetime(2025, 1, 17, 10, 0),  # New date
            ],
            "value": ["d", "e"],
        })
        combined = pl.concat([initial_data, new_data])
        combined.write_parquet(data_file)

        # Second run: should append only 2 new rows
        app.run(silent=True)

        # Verify final state - SHOULD be 5, but bug causes 8
        result2 = sink.read("events").collect()
        assert len(result2) == 5, f"Expected 5 rows after second run, got {len(result2)}"
