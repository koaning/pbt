"""General sink tests that run against all sink implementations."""

import datetime
from pathlib import Path

import polars as pl
import pytest

from pbt.sinks import DuckDBSink, LocalSink, Sink


@pytest.fixture
def local_sink(tmp_path: Path) -> LocalSink:
    """Create a LocalSink with a temporary directory."""
    return LocalSink(path=tmp_path / "output")


@pytest.fixture
def duckdb_sink(tmp_path: Path) -> DuckDBSink:
    """Create a DuckDBSink with a temporary file."""
    sink = DuckDBSink(path=str(tmp_path / "test.db"))
    yield sink
    sink.close()


@pytest.fixture(params=["local", "duckdb"])
def sink(request, local_sink: LocalSink, duckdb_sink: DuckDBSink) -> Sink:
    """Parametrized fixture that returns each sink implementation."""
    if request.param == "local":
        return local_sink
    else:
        return duckdb_sink


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Sample DataFrame for testing."""
    return pl.DataFrame(
        {"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10, 20, 30]}
    )


@pytest.fixture
def partitioned_df() -> pl.DataFrame:
    """Sample DataFrame with a date column for partition testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date": [
                datetime.date(2025, 1, 1),
                datetime.date(2025, 1, 1),
                datetime.date(2025, 1, 2),
                datetime.date(2025, 1, 2),
            ],
            "value": [10, 20, 30, 40],
        }
    )


# Write/read roundtrip tests


def test_simple_write_read(sink: Sink, sample_df: pl.DataFrame):
    """Write and read back a simple table."""
    result = sink.write(sample_df, "test_table")

    assert result.rows_written == 3
    assert result.operation in ("create", "append")

    read_df = sink.read("test_table").collect()
    assert len(read_df) == 3
    assert set(read_df.columns) == {"id", "name", "value"}
    assert read_df["id"].to_list() == [1, 2, 3]


def test_partitioned_write_read(sink: Sink, partitioned_df: pl.DataFrame):
    """Write and read back a partitioned table."""
    result = sink.write(partitioned_df, "events", partition_by=["date"])

    assert result.rows_written == 4
    assert len(result.partitions_affected) == 2

    read_df = sink.read("events").collect()
    assert len(read_df) == 4
    assert "date" in read_df.columns


def test_partition_filter_read(sink: Sink, partitioned_df: pl.DataFrame):
    """Read with partition filter."""
    sink.write(partitioned_df, "events", partition_by=["date"])

    read_df = sink.read(
        "events", partition_filter={"date": datetime.date(2025, 1, 1)}
    ).collect()
    assert len(read_df) == 2
    assert all(d == datetime.date(2025, 1, 1) for d in read_df["date"].to_list())


# Exists tests


def test_exists_false_for_missing(sink: Sink):
    """exists() returns False for non-existent table."""
    assert sink.exists("nonexistent") is False


def test_exists_true_after_write(sink: Sink, sample_df: pl.DataFrame):
    """exists() returns True after writing."""
    sink.write(sample_df, "test_table")
    assert sink.exists("test_table") is True


# Partition tests


def test_list_partitions_empty_for_non_partitioned(sink: Sink, sample_df: pl.DataFrame):
    """list_partitions() returns empty for non-partitioned table."""
    sink.write(sample_df, "test_table")
    assert sink.list_partitions("test_table") == []


def test_list_partitions_returns_hive_style(sink: Sink, partitioned_df: pl.DataFrame):
    """list_partitions() returns Hive-style partition paths."""
    sink.write(partitioned_df, "events", partition_by=["date"])

    partitions = sink.list_partitions("events")
    assert len(partitions) == 2
    assert all("date=" in p for p in partitions)
    assert "date=2025-01-01" in partitions
    assert "date=2025-01-02" in partitions


def test_delete_partitions(sink: Sink, partitioned_df: pl.DataFrame):
    """delete_partitions() removes specified partitions."""
    sink.write(partitioned_df, "events", partition_by=["date"])

    sink.delete_partitions("events", ["date=2025-01-01"])

    partitions = sink.list_partitions("events")
    assert partitions == ["date=2025-01-02"]

    read_df = sink.read("events").collect()
    assert len(read_df) == 2
    assert all(d == datetime.date(2025, 1, 2) for d in read_df["date"].to_list())


# Partition mode tests


def test_append_mode_adds_data(sink: Sink):
    """Append mode adds data to existing partitions."""
    df1 = pl.DataFrame(
        {
            "id": [1, 2],
            "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
        }
    )
    df2 = pl.DataFrame(
        {
            "id": [3, 4],
            "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
        }
    )

    sink.write(df1, "events", partition_by=["date"], partition_mode="append")
    sink.write(df2, "events", partition_by=["date"], partition_mode="append")

    read_df = sink.read("events").collect()
    assert len(read_df) == 4
    assert set(read_df["id"].to_list()) == {1, 2, 3, 4}


def test_overwrite_mode_replaces_partition(sink: Sink):
    """Overwrite mode replaces data in existing partitions."""
    df1 = pl.DataFrame(
        {
            "id": [1, 2],
            "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
        }
    )
    df2 = pl.DataFrame(
        {
            "id": [3, 4],
            "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
        }
    )

    sink.write(df1, "events", partition_by=["date"], partition_mode="append")
    sink.write(df2, "events", partition_by=["date"], partition_mode="overwrite")

    read_df = sink.read("events").collect()
    assert len(read_df) == 2
    assert set(read_df["id"].to_list()) == {3, 4}


# Dry run tests


def test_dry_run_returns_plan(sink: Sink, sample_df: pl.DataFrame):
    """dry_run=True returns plan without writing."""
    plan = sink.write(sample_df, "test_table", dry_run=True)

    assert plan.table_name == "test_table"
    assert plan.rows_to_write == 3
    assert sink.exists("test_table") is False


def test_dry_run_shows_partition_operations(sink: Sink, partitioned_df: pl.DataFrame):
    """dry_run shows planned partition operations."""
    plan = sink.write(partitioned_df, "events", partition_by=["date"], dry_run=True)

    assert len(plan.partitions_affected) == 2
    assert all(op == "create" for op in plan.partition_operations.values())
