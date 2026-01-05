"""Tests for PBT validation and error handling."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from pbt import LocalSink, conf
from pbt.exceptions import CircularDependencyError, IncrementalConfigError


# --- Circular Dependency Detection ---


def test_simple_cycle_raises_error(tmp_path):
    """A -> B -> A cycle is detected."""
    app = conf(root=tmp_path)

    @app.source
    def base():
        return pl.LazyFrame({"x": [1, 2, 3]})

    @app.model
    def model_a(model_b):
        return model_b.select("x")

    @app.model
    def model_b(model_a):
        return model_a.select("x")

    with pytest.raises(CircularDependencyError) as exc_info:
        app.run(silent=True)

    assert "Circular dependency detected" in str(exc_info.value)
    assert "model_a" in str(exc_info.value)
    assert "model_b" in str(exc_info.value)


def test_self_referencing_cycle(tmp_path):
    """Self-referencing model is detected."""
    app = conf(root=tmp_path)

    @app.model
    def self_ref(self_ref):
        return self_ref.select("x")

    with pytest.raises(CircularDependencyError):
        app.run(silent=True)


def test_no_cycle_succeeds(tmp_path):
    """Valid DAG executes without error."""
    app = conf(root=tmp_path)

    @app.source
    def base():
        return pl.LazyFrame({"x": [1, 2, 3]})

    @app.model
    def step1(base):
        return base.select("x")

    @app.model
    def step2(step1):
        return step1.select("x")

    result = app.run(silent=True)
    assert "step2" in result


# --- Time Column Validation ---


def test_missing_time_column_raises_error(tmp_path):
    """Missing time_column raises IncrementalConfigError."""
    app = conf(root=tmp_path)
    sink = LocalSink(path=tmp_path / "output")

    @app.source
    def events():
        return pl.LazyFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})

    @app.incremental_table(
        time_column="timestamp",  # doesn't exist
        partition_by="date",
        sink=sink,
    )
    def processed(events):
        return events.with_columns(pl.lit(date(2025, 1, 15)).alias("date"))

    with pytest.raises(IncrementalConfigError) as exc_info:
        app.run(silent=True)

    assert "time_column 'timestamp' not found" in str(exc_info.value)
    assert "Available columns" in str(exc_info.value)


def test_non_datetime_time_column_raises_error(tmp_path):
    """Non-datetime time_column raises IncrementalConfigError."""
    app = conf(root=tmp_path)
    sink = LocalSink(path=tmp_path / "output")

    @app.source
    def events():
        return pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "timestamp": ["2025-01-15", "2025-01-16", "2025-01-17"],  # String
            }
        )

    @app.incremental_table(
        time_column="timestamp",
        partition_by="date",
        sink=sink,
    )
    def processed(events):
        return events.with_columns(pl.lit(date(2025, 1, 15)).alias("date"))

    with pytest.raises(IncrementalConfigError) as exc_info:
        app.run(silent=True)

    assert "must be Datetime or Date type" in str(exc_info.value)
    assert "Hint:" in str(exc_info.value)


def test_valid_datetime_time_column_succeeds(tmp_path):
    """Valid datetime time_column passes validation."""
    app = conf(root=tmp_path)
    sink = LocalSink(path=tmp_path / "output")

    @app.source
    def events():
        return pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "timestamp": [
                    datetime(2025, 1, 15, 10, 0),
                    datetime(2025, 1, 16, 10, 0),
                    datetime(2025, 1, 17, 10, 0),
                ],
            }
        )

    @app.incremental_table(
        time_column="timestamp",
        partition_by="date",
        sink=sink,
    )
    def processed(events):
        return events.with_columns(pl.col("timestamp").dt.date().alias("date"))

    app.run(silent=True)
    result = sink.read("processed").collect()
    assert len(result) == 3


# --- Date Partition Requirement ---


def test_missing_date_partition_raises_error(tmp_path):
    """Non-date partition raises IncrementalConfigError."""
    app = conf(root=tmp_path)
    sink = LocalSink(path=tmp_path / "output")

    @app.source
    def events():
        return pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "timestamp": [
                    datetime(2025, 1, 15, 10, 0),
                    datetime(2025, 1, 16, 10, 0),
                    datetime(2025, 1, 17, 10, 0),
                ],
                "region": ["US", "EU", "US"],
            }
        )

    @app.incremental_table(
        time_column="timestamp",
        partition_by="region",  # Not a date
        sink=sink,
    )
    def processed(events):
        return events

    with pytest.raises(IncrementalConfigError) as exc_info:
        app.run(silent=True)

    assert "requires at least one Date partition" in str(exc_info.value)
    assert "Hint:" in str(exc_info.value)


def test_date_partition_succeeds(tmp_path):
    """Date partition passes validation."""
    app = conf(root=tmp_path)
    sink = LocalSink(path=tmp_path / "output")

    @app.source
    def events():
        return pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "timestamp": [
                    datetime(2025, 1, 15, 10, 0),
                    datetime(2025, 1, 16, 10, 0),
                    datetime(2025, 1, 17, 10, 0),
                ],
            }
        )

    @app.incremental_table(
        time_column="timestamp",
        partition_by="date",
        sink=sink,
    )
    def processed(events):
        return events.with_columns(pl.col("timestamp").dt.date().alias("date"))

    app.run(silent=True)


def test_mixed_partitions_with_date_succeeds(tmp_path):
    """Mixed partitions including date pass validation."""
    app = conf(root=tmp_path)
    sink = LocalSink(path=tmp_path / "output")

    @app.source
    def events():
        return pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "timestamp": [
                    datetime(2025, 1, 15, 10, 0),
                    datetime(2025, 1, 16, 10, 0),
                    datetime(2025, 1, 17, 10, 0),
                ],
                "region": ["US", "EU", "US"],
            }
        )

    @app.incremental_table(
        time_column="timestamp",
        partition_by=["region", "date"],
        sink=sink,
    )
    def processed(events):
        return events.with_columns(pl.col("timestamp").dt.date().alias("date"))

    app.run(silent=True)


# --- Lookback Feature ---


def test_lookback_reprocesses_window(tmp_path):
    """Lookback reprocesses the specified time window."""
    output_dir = tmp_path / "output"
    data_file = tmp_path / "events.parquet"

    initial_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "timestamp": [
                datetime(2025, 1, 15, 10, 0),
                datetime(2025, 1, 16, 10, 0),
                datetime(2025, 1, 17, 10, 0),
            ],
            "value": ["a", "b", "c"],
        }
    )
    initial_data.write_parquet(data_file)

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
        lookback=timedelta(days=1),
        sink=sink,
    )
    def events(raw_events):
        return raw_events

    # First run
    app.run(silent=True)
    result1 = sink.read("events").collect()
    assert len(result1) == 3

    # Add late-arriving data for Jan 16 (within lookback window)
    new_data = pl.DataFrame(
        {
            "id": [4],
            "timestamp": [datetime(2025, 1, 16, 15, 0)],
            "value": ["d"],
        }
    )
    combined = pl.concat([initial_data, new_data])
    combined.write_parquet(data_file)

    # Second run - lookback should catch the late event
    app.run(silent=True)
    result2 = sink.read("events").collect()

    assert len(result2) == 4

    jan16_events = result2.filter(pl.col("date") == date(2025, 1, 16))
    assert len(jan16_events) == 2  # Original + late event


def test_lookback_uses_overwrite_mode(tmp_path):
    """Lookback uses overwrite mode to prevent duplicates."""
    output_dir = tmp_path / "output"
    data_file = tmp_path / "events.parquet"

    data = pl.DataFrame(
        {
            "id": [1, 2],
            "timestamp": [
                datetime(2025, 1, 15, 10, 0),
                datetime(2025, 1, 16, 10, 0),
            ],
            "value": ["a", "b"],
        }
    )
    data.write_parquet(data_file)

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
        lookback=timedelta(days=2),
        sink=sink,
    )
    def events(raw_events):
        return raw_events

    # Run twice with same data
    app.run(silent=True)
    app.run(silent=True)

    result = sink.read("events").collect()
    assert len(result) == 2  # No duplicates
