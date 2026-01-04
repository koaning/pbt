"""Test for incremental + partitioned append bug."""
from datetime import date, datetime

import polars as pl

from pbt import LocalSink, conf


def test_incremental_partitioned_append_no_duplicates(tmp_path):
    """
    Bug: When using incremental_table with partition_by + partition_mode="append",
    data gets duplicated because:
    1. core._apply_incremental_merge() reads existing + concats new
    2. sink._write_partitioned() ALSO appends to existing partition files

    Expected: 5 rows total after two runs
    Actual (with bug): 8 rows (3 initial + 5 on second run due to double-append)
    """
    output_dir = tmp_path / "output"

    # Create sample data file with timestamps
    data_file = tmp_path / "events.parquet"
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
    app = conf(root=tmp_path)
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

    # Verify partitions created (2 dates: 2025-01-15 and 2025-01-16)
    partitions1 = sink.list_partitions("events")
    assert len(partitions1) == 2, f"Expected 2 partitions after first run, got {len(partitions1)}"

    # Add more data with newer timestamps (one existing date, one new date)
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

    # Verify new partition was created (now 3 dates)
    partitions2 = sink.list_partitions("events")
    assert len(partitions2) == 3, f"Expected 3 partitions after second run, got {len(partitions2)}"

    # Verify the new date partition exists
    assert any("2025-01-17" in p for p in partitions2), "Expected partition for 2025-01-17"

    # Verify row counts per partition
    for partition_date, expected_count in [(date(2025, 1, 15), 2), (date(2025, 1, 16), 2), (date(2025, 1, 17), 1)]:
        partition_data = sink.read("events", partition_filter={"date": partition_date}).collect()
        assert len(partition_data) == expected_count, (
            f"Expected {expected_count} rows for {partition_date}, got {len(partition_data)}"
        )
