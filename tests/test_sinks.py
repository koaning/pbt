"""General sink tests that run against all sink implementations."""

import datetime
from datetime import date, datetime as dt
from pathlib import Path

import polars as pl
import pytest

from pbt import conf
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
# Incremental table simulation tests (parameterized)
# =============================================================================

# Test incremental append workflow with various partition types
INCREMENTAL_CASES = [
    pytest.param(
        # Initial data
        pl.DataFrame({
            "id": [1, 2, 3],
            "date": [
                datetime.date(2025, 1, 15),
                datetime.date(2025, 1, 15),
                datetime.date(2025, 1, 16),
            ],
            "value": ["a", "b", "c"],
        }),
        # New data to append
        pl.DataFrame({
            "id": [4, 5],
            "date": [
                datetime.date(2025, 1, 16),
                datetime.date(2025, 1, 17),
            ],
            "value": ["d", "e"],
        }),
        ["date"],
        3,  # expected partitions after append
        id="date_partition",
    ),
    pytest.param(
        # Initial data with year/month partitions
        pl.DataFrame({
            "id": [1, 2, 3],
            "year": [2024, 2024, 2025],
            "month": [12, 12, 1],
            "value": ["a", "b", "c"],
        }),
        # New data to append
        pl.DataFrame({
            "id": [4, 5],
            "year": [2025, 2025],
            "month": [1, 2],
            "value": ["d", "e"],
        }),
        ["year", "month"],
        3,  # expected partitions: 2024/12, 2025/1, 2025/2
        id="year_month_partition",
    ),
    pytest.param(
        # Initial data with category partition
        pl.DataFrame({
            "id": [1, 2, 3],
            "category": ["users", "users", "orders"],
            "value": ["a", "b", "c"],
        }),
        # New data to append
        pl.DataFrame({
            "id": [4, 5],
            "category": ["orders", "products"],
            "value": ["d", "e"],
        }),
        ["category"],
        3,  # expected partitions: users, orders, products
        id="string_partition",
    ),
]


@pytest.mark.parametrize(
    "initial_df,new_df,partition_by,expected_partitions", INCREMENTAL_CASES
)
def test_incremental_append_workflow(
    sink: Sink,
    initial_df: pl.DataFrame,
    new_df: pl.DataFrame,
    partition_by: list[str],
    expected_partitions: int,
):
    """
    Simulate incremental table workflow with various partition types.

    Tests the append pattern:
    1. Write initial data partitioned by columns
    2. Append new data (some to existing partitions, some to new)
    3. Verify correct total rows and partition counts
    """
    initial_rows = len(initial_df)
    new_rows = len(new_df)

    # Initial write
    sink.write(initial_df, "events", partition_by=partition_by, partition_mode="append")
    result1 = sink.read("events").collect()
    assert len(result1) == initial_rows

    # Append new data
    sink.write(new_df, "events", partition_by=partition_by, partition_mode="append")

    # Verify final state
    result2 = sink.read("events").collect()
    assert len(result2) == initial_rows + new_rows
    assert len(sink.list_partitions("events")) == expected_partitions

    # Verify all IDs are present (no duplicates, no missing)
    all_ids = set(initial_df["id"].to_list() + new_df["id"].to_list())
    assert set(result2["id"].to_list()) == all_ids


# Test overwrite behavior with various partition types
OVERWRITE_CASES = [
    pytest.param(
        pl.DataFrame({
            "id": [1, 2, 3, 4],
            "date": [
                datetime.date(2025, 1, 15),
                datetime.date(2025, 1, 15),
                datetime.date(2025, 1, 16),
                datetime.date(2025, 1, 16),
            ],
            "value": [10, 20, 30, 40],
        }),
        pl.DataFrame({
            "id": [3, 4],
            "date": [datetime.date(2025, 1, 16), datetime.date(2025, 1, 16)],
            "value": [300, 400],
        }),
        ["date"],
        {"date": datetime.date(2025, 1, 16)},
        {300, 400},
        id="date_partition_overwrite",
    ),
    pytest.param(
        pl.DataFrame({
            "id": [1, 2, 3, 4],
            "year": [2024, 2024, 2025, 2025],
            "month": [12, 12, 1, 1],
            "value": [10, 20, 30, 40],
        }),
        pl.DataFrame({
            "id": [3, 4],
            "year": [2025, 2025],
            "month": [1, 1],
            "value": [300, 400],
        }),
        ["year", "month"],
        {"year": 2025, "month": 1},
        {300, 400},
        id="year_month_partition_overwrite",
    ),
]


@pytest.mark.parametrize(
    "initial_df,overwrite_df,partition_by,filter_dict,expected_values", OVERWRITE_CASES
)
def test_incremental_overwrite_workflow(
    sink: Sink,
    initial_df: pl.DataFrame,
    overwrite_df: pl.DataFrame,
    partition_by: list[str],
    filter_dict: dict,
    expected_values: set,
):
    """
    Simulate lookback/overwrite behavior with various partition types.

    Tests that overwrite mode correctly replaces only the targeted partition.
    """
    # Initial write
    sink.write(initial_df, "events", partition_by=partition_by, partition_mode="append")

    # Overwrite specific partition
    sink.write(
        overwrite_df, "events", partition_by=partition_by, partition_mode="overwrite"
    )

    # Verify overwritten partition has new values
    overwritten = sink.read("events", partition_filter=filter_dict).collect()
    assert set(overwritten["value"].to_list()) == expected_values


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


# =============================================================================
# Integration tests with PBT incremental_table decorator
# =============================================================================


@pytest.fixture
def make_sink(tmp_path: Path):
    """Factory fixture that creates sinks for integration tests."""
    sinks = []

    def _make(sink_type: str) -> Sink:
        if sink_type == "local":
            sink = LocalSink(path=tmp_path / "output")
        else:
            sink = DuckDBSink(path=str(tmp_path / "test.db"))
        sinks.append(sink)
        return sink

    yield _make

    for s in sinks:
        if hasattr(s, "close"):
            s.close()


@pytest.mark.parametrize("sink_type", ["local", "duckdb"])
def test_incremental_table_with_sink(tmp_path: Path, make_sink, sink_type: str):
    """
    Integration test: incremental_table decorator with various sink types.

    This tests the REAL incremental_table workflow, not just sink operations.
    Verifies that the full PBT pipeline works with DuckDB sink.
    """
    sink = make_sink(sink_type)

    # Create sample data file with timestamps
    data_file = tmp_path / "events.parquet"
    initial_data = pl.DataFrame({
        "id": [1, 2, 3],
        "timestamp": [
            dt(2025, 1, 15, 10, 0),
            dt(2025, 1, 15, 11, 0),
            dt(2025, 1, 16, 10, 0),
        ],
        "value": ["a", "b", "c"],
    })
    initial_data.write_parquet(data_file)

    # Setup PBT app
    app = conf(root=tmp_path)

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

    result1 = sink.read("events").collect()
    assert len(result1) == 3, f"Expected 3 rows after first run, got {len(result1)}"

    partitions1 = sink.list_partitions("events")
    assert len(partitions1) == 2, f"Expected 2 partitions, got {len(partitions1)}"

    # Add more data with newer timestamps
    new_data = pl.DataFrame({
        "id": [4, 5],
        "timestamp": [
            dt(2025, 1, 16, 12, 0),  # Same date as id=3
            dt(2025, 1, 17, 10, 0),  # New date
        ],
        "value": ["d", "e"],
    })
    combined = pl.concat([initial_data, new_data])
    combined.write_parquet(data_file)

    # Second run: should append only 2 new rows
    app.run(silent=True)

    result2 = sink.read("events").collect()
    assert len(result2) == 5, f"Expected 5 rows after second run, got {len(result2)}"

    partitions2 = sink.list_partitions("events")
    assert len(partitions2) == 3, f"Expected 3 partitions, got {len(partitions2)}"

    # Verify per-partition counts
    for partition_date, expected_count in [
        (date(2025, 1, 15), 2),
        (date(2025, 1, 16), 2),
        (date(2025, 1, 17), 1),
    ]:
        partition_data = sink.read(
            "events", partition_filter={"date": partition_date}
        ).collect()
        assert len(partition_data) == expected_count, (
            f"Expected {expected_count} rows for {partition_date}"
        )
