"""Tests for PBT validation and error handling."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from pbt import LocalSink, conf
from pbt.exceptions import CircularDependencyError, IncrementalConfigError


class TestCircularDependencyDetection:
    """Test that circular dependencies are properly detected."""

    def test_simple_cycle_raises_error(self, tmp_path):
        """Test that a simple A -> B -> A cycle is detected."""
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
        # Should mention the cycle path
        assert "model_a" in str(exc_info.value)
        assert "model_b" in str(exc_info.value)

    def test_self_referencing_cycle(self, tmp_path):
        """Test that a self-referencing model is detected."""
        app = conf(root=tmp_path)

        @app.model
        def self_ref(self_ref):
            return self_ref.select("x")

        with pytest.raises(CircularDependencyError):
            app.run(silent=True)

    def test_no_cycle_succeeds(self, tmp_path):
        """Test that a valid DAG executes without error."""
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

        # Should not raise
        result = app.run(silent=True)
        assert "step2" in result


class TestTimeColumnValidation:
    """Test that time_column validation works correctly."""

    def test_missing_time_column_raises_error(self, tmp_path):
        """Test that a missing time_column raises IncrementalConfigError."""
        app = conf(root=tmp_path)
        sink = LocalSink(path=tmp_path / "output")

        @app.source
        def events():
            return pl.LazyFrame(
                {
                    "id": [1, 2, 3],
                    "value": ["a", "b", "c"],
                }
            )

        @app.incremental_table(
            time_column="timestamp",  # This column doesn't exist
            partition_by="date",
            sink=sink,
        )
        def processed(events):
            return events.with_columns(pl.lit(date(2025, 1, 15)).alias("date"))

        with pytest.raises(IncrementalConfigError) as exc_info:
            app.run(silent=True)

        assert "time_column 'timestamp' not found" in str(exc_info.value)
        assert "Available columns" in str(exc_info.value)

    def test_non_datetime_time_column_raises_error(self, tmp_path):
        """Test that a non-datetime time_column raises IncrementalConfigError."""
        app = conf(root=tmp_path)
        sink = LocalSink(path=tmp_path / "output")

        @app.source
        def events():
            return pl.LazyFrame(
                {
                    "id": [1, 2, 3],
                    "timestamp": [
                        "2025-01-15",
                        "2025-01-16",
                        "2025-01-17",
                    ],  # String, not datetime
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

    def test_valid_datetime_time_column_succeeds(self, tmp_path):
        """Test that a valid datetime time_column passes validation."""
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

        # Should not raise
        app.run(silent=True)
        result = sink.read("processed").collect()
        assert len(result) == 3


class TestDatePartitionRequirement:
    """Test that incremental tables require a date partition."""

    def test_missing_date_partition_raises_error(self, tmp_path):
        """Test that a non-date partition raises IncrementalConfigError."""
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
            partition_by="region",  # Not a date partition
            sink=sink,
        )
        def processed(events):
            return events

        with pytest.raises(IncrementalConfigError) as exc_info:
            app.run(silent=True)

        assert "requires at least one Date partition" in str(exc_info.value)
        assert "Hint:" in str(exc_info.value)

    def test_date_partition_succeeds(self, tmp_path):
        """Test that a date partition passes validation."""
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

        # Should not raise
        app.run(silent=True)

    def test_mixed_partitions_with_date_succeeds(self, tmp_path):
        """Test that mixed partitions including date pass validation."""
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
            partition_by=["region", "date"],  # Mixed: string + date
            sink=sink,
        )
        def processed(events):
            return events.with_columns(pl.col("timestamp").dt.date().alias("date"))

        # Should not raise
        app.run(silent=True)


class TestLookbackFeature:
    """Test the lookback feature for late-arriving data."""

    def test_lookback_reprocesses_window(self, tmp_path):
        """Test that lookback reprocesses the specified time window."""
        output_dir = tmp_path / "output"
        data_file = tmp_path / "events.parquet"

        # Create initial data
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
            lookback=timedelta(days=1),  # Look back 1 day
            sink=sink,
        )
        def events(raw_events):
            return raw_events

        # First run
        app.run(silent=True)
        result1 = sink.read("events").collect()
        assert len(result1) == 3

        # Add late-arriving data for Jan 16 (within lookback window from Jan 17)
        new_data = pl.DataFrame(
            {
                "id": [4],
                "timestamp": [datetime(2025, 1, 16, 15, 0)],  # Late event on Jan 16
                "value": ["d"],
            }
        )
        combined = pl.concat([initial_data, new_data])
        combined.write_parquet(data_file)

        # Second run - lookback should catch the late event
        app.run(silent=True)
        result2 = sink.read("events").collect()

        # Should have all 4 events
        assert len(result2) == 4

        # Verify the late event is present
        jan16_events = result2.filter(pl.col("date") == date(2025, 1, 16))
        assert len(jan16_events) == 2  # Original + late event

    def test_lookback_uses_overwrite_mode(self, tmp_path):
        """Test that lookback uses overwrite mode to prevent duplicates."""
        output_dir = tmp_path / "output"
        data_file = tmp_path / "events.parquet"

        # Create data
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

        # Should not have duplicates
        result = sink.read("events").collect()
        assert len(result) == 2  # No duplicates
