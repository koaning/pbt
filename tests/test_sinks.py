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


# =============================================================================
# Parameterized tests for various data types and shapes
# =============================================================================

# Test various DataFrame schemas across all sinks
DATAFRAME_CASES = [
    pytest.param(
        pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
        id="int_and_str",
    ),
    pytest.param(
        pl.DataFrame({"x": [1.5, 2.5, 3.5], "y": [True, False, True]}),
        id="float_and_bool",
    ),
    pytest.param(
        pl.DataFrame({
            "dt": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)],
            "val": [100, 200],
        }),
        id="date_and_int",
    ),
    pytest.param(
        pl.DataFrame({"single": [42]}),
        id="single_row",
    ),
    pytest.param(
        pl.DataFrame({"a": list(range(1000)), "b": list(range(1000))}),
        id="large_1000_rows",
    ),
]


@pytest.mark.parametrize("df", DATAFRAME_CASES)
def test_write_read_various_schemas(sink: Sink, df: pl.DataFrame):
    """Write and read back DataFrames with various schemas."""
    sink.write(df, "test_table")
    result = sink.read("test_table").collect()

    assert len(result) == len(df)
    assert set(result.columns) == set(df.columns)


# Test various partition configurations
PARTITION_CASES = [
    pytest.param(
        pl.DataFrame({
            "id": [1, 2, 3, 4],
            "date": [
                datetime.date(2025, 1, 1),
                datetime.date(2025, 1, 1),
                datetime.date(2025, 1, 2),
                datetime.date(2025, 1, 2),
            ],
        }),
        ["date"],
        2,
        id="single_date_partition",
    ),
    pytest.param(
        pl.DataFrame({
            "id": [1, 2, 3, 4],
            "year": [2024, 2024, 2025, 2025],
            "month": [1, 2, 1, 2],
        }),
        ["year", "month"],
        4,
        id="two_column_partition",
    ),
    pytest.param(
        pl.DataFrame({
            "id": [1, 2, 3],
            "category": ["a", "b", "c"],
        }),
        ["category"],
        3,
        id="string_partition",
    ),
]


@pytest.mark.parametrize("df,partition_by,expected_partitions", PARTITION_CASES)
def test_partitioning_configurations(
    sink: Sink, df: pl.DataFrame, partition_by: list[str], expected_partitions: int
):
    """Test various partitioning configurations."""
    sink.write(df, "partitioned", partition_by=partition_by)

    partitions = sink.list_partitions("partitioned")
    assert len(partitions) == expected_partitions

    # All partitions should be Hive-style
    for col in partition_by:
        assert all(f"{col}=" in p for p in partitions)

    # Read back should have all rows
    result = sink.read("partitioned").collect()
    assert len(result) == len(df)


# Test partition modes
PARTITION_MODE_CASES = [
    pytest.param("append", 4, {1, 2, 3, 4}, id="append_keeps_all"),
    pytest.param("overwrite", 2, {3, 4}, id="overwrite_replaces"),
]


@pytest.mark.parametrize("mode,expected_rows,expected_ids", PARTITION_MODE_CASES)
def test_partition_modes(sink: Sink, mode: str, expected_rows: int, expected_ids: set):
    """Test append vs overwrite partition modes."""
    df1 = pl.DataFrame({
        "id": [1, 2],
        "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
    })
    df2 = pl.DataFrame({
        "id": [3, 4],
        "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
    })

    sink.write(df1, "events", partition_by=["date"], partition_mode="append")
    sink.write(df2, "events", partition_by=["date"], partition_mode=mode)

    result = sink.read("events").collect()
    assert len(result) == expected_rows
    assert set(result["id"].to_list()) == expected_ids


# =============================================================================
# Incremental table simulation tests
# =============================================================================


def test_incremental_table_workflow(sink: Sink):
    """
    Simulate incremental table workflow: initial write + incremental append.

    This tests the pattern used by incremental_table decorator:
    1. Write initial data partitioned by date
    2. Append new data to existing and new partitions
    3. Verify no duplicates and correct partition counts
    """
    from datetime import datetime as dt

    # Initial data: 3 rows across 2 dates
    initial = pl.DataFrame({
        "id": [1, 2, 3],
        "timestamp": [
            dt(2025, 1, 15, 10, 0),
            dt(2025, 1, 15, 11, 0),
            dt(2025, 1, 16, 10, 0),
        ],
        "value": ["a", "b", "c"],
    }).with_columns(pl.col("timestamp").dt.date().alias("date"))

    sink.write(initial, "events", partition_by=["date"], partition_mode="append")

    # Verify initial state
    result1 = sink.read("events").collect()
    assert len(result1) == 3
    assert len(sink.list_partitions("events")) == 2

    # New data: 2 rows - one existing date, one new date
    new_data = pl.DataFrame({
        "id": [4, 5],
        "timestamp": [
            dt(2025, 1, 16, 12, 0),  # Same date as id=3
            dt(2025, 1, 17, 10, 0),  # New date
        ],
        "value": ["d", "e"],
    }).with_columns(pl.col("timestamp").dt.date().alias("date"))

    sink.write(new_data, "events", partition_by=["date"], partition_mode="append")

    # Verify final state
    result2 = sink.read("events").collect()
    assert len(result2) == 5, f"Expected 5 rows, got {len(result2)}"
    assert len(sink.list_partitions("events")) == 3

    # Verify per-partition counts
    for partition_date, expected_count in [
        (datetime.date(2025, 1, 15), 2),
        (datetime.date(2025, 1, 16), 2),
        (datetime.date(2025, 1, 17), 1),
    ]:
        partition_data = sink.read(
            "events", partition_filter={"date": partition_date}
        ).collect()
        assert len(partition_data) == expected_count


def test_incremental_with_lookback_overwrite(sink: Sink):
    """
    Simulate lookback behavior: overwrite partitions within lookback window.

    When lookback is set, incremental tables use overwrite mode for
    partitions that fall within the lookback window.
    """
    # Initial data
    initial = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "date": [
            datetime.date(2025, 1, 15),
            datetime.date(2025, 1, 15),
            datetime.date(2025, 1, 16),
            datetime.date(2025, 1, 16),
        ],
        "value": [10, 20, 30, 40],
    })
    sink.write(initial, "events", partition_by=["date"], partition_mode="append")

    # Lookback reprocess: overwrite 2025-01-16 partition with corrected data
    corrected = pl.DataFrame({
        "id": [3, 4],
        "date": [datetime.date(2025, 1, 16), datetime.date(2025, 1, 16)],
        "value": [300, 400],  # Corrected values
    })
    sink.write(corrected, "events", partition_by=["date"], partition_mode="overwrite")

    # Verify: 2025-01-15 unchanged, 2025-01-16 replaced
    result = sink.read("events").collect()
    assert len(result) == 4

    jan15 = sink.read(
        "events", partition_filter={"date": datetime.date(2025, 1, 15)}
    ).collect()
    assert len(jan15) == 2
    assert set(jan15["value"].to_list()) == {10, 20}

    jan16 = sink.read(
        "events", partition_filter={"date": datetime.date(2025, 1, 16)}
    ).collect()
    assert len(jan16) == 2
    assert set(jan16["value"].to_list()) == {300, 400}


# =============================================================================
# Core functionality tests
# =============================================================================


def test_exists_false_for_missing(sink: Sink):
    """exists() returns False for non-existent table."""
    assert sink.exists("nonexistent") is False


def test_exists_true_after_write(sink: Sink):
    """exists() returns True after writing."""
    df = pl.DataFrame({"id": [1, 2, 3]})
    sink.write(df, "test_table")
    assert sink.exists("test_table") is True


def test_list_partitions_empty_for_non_partitioned(sink: Sink):
    """list_partitions() returns empty for non-partitioned table."""
    df = pl.DataFrame({"id": [1, 2, 3]})
    sink.write(df, "test_table")
    assert sink.list_partitions("test_table") == []


def test_delete_partitions(sink: Sink):
    """delete_partitions() removes specified partitions."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "date": [
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 2),
            datetime.date(2025, 1, 2),
        ],
    })
    sink.write(df, "events", partition_by=["date"])
    sink.delete_partitions("events", ["date=2025-01-01"])

    partitions = sink.list_partitions("events")
    assert partitions == ["date=2025-01-02"]

    result = sink.read("events").collect()
    assert len(result) == 2


def test_partition_filter_read(sink: Sink):
    """Read with partition filter returns only matching rows."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "date": [
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 2),
            datetime.date(2025, 1, 2),
        ],
    })
    sink.write(df, "events", partition_by=["date"])

    result = sink.read(
        "events", partition_filter={"date": datetime.date(2025, 1, 1)}
    ).collect()
    assert len(result) == 2
    assert all(d == datetime.date(2025, 1, 1) for d in result["date"].to_list())


def test_dry_run_returns_plan_without_writing(sink: Sink):
    """dry_run=True returns plan without actually writing."""
    df = pl.DataFrame({"id": [1, 2, 3]})
    plan = sink.write(df, "test_table", dry_run=True)

    assert plan.table_name == "test_table"
    assert plan.rows_to_write == 3
    assert sink.exists("test_table") is False


def test_dry_run_shows_partition_operations(sink: Sink):
    """dry_run shows planned partition operations."""
    df = pl.DataFrame({
        "id": [1, 2],
        "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)],
    })
    plan = sink.write(df, "events", partition_by=["date"], dry_run=True)

    assert len(plan.partitions_affected) == 2
    assert all(op == "create" for op in plan.partition_operations.values())
