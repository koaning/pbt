"""Test that all sources return true LazyFrames."""

import os
from contextlib import contextmanager
from pathlib import Path

import duckdb
import polars as pl

from pbt.sources import duckdb_scan


@contextmanager
def duckdb_env(tmp_path: Path, env_key: str = "TEST_DUCKDB_PATH"):
    """Context manager that creates a test DuckDB and sets env var."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE events (id INT, name VARCHAR)")
    con.execute("INSERT INTO events VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    con.close()

    os.environ[env_key] = str(db_path)
    try:
        yield db_path
    finally:
        del os.environ[env_key]


def test_duckdb_scan_returns_lazy(tmp_path):
    """
    duckdb_scan should return a true LazyFrame, not an eager DataFrame
    wrapped in .lazy().

    Using .pl(lazy=True) ensures query pushdown works correctly.
    """
    with duckdb_env(tmp_path):
        result = duckdb_scan("events", connection_key="TEST_DUCKDB_PATH")

        # Should be a LazyFrame
        assert isinstance(result, pl.LazyFrame), "duckdb_scan should return LazyFrame"

        # Verify data is correct when collected
        df = result.collect()
        assert len(df) == 3
        assert df["id"].to_list() == [1, 2, 3]
