"""Test that all sources return true LazyFrames."""
import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

# Only run if duckdb is installed
duckdb = pytest.importorskip("duckdb")

from pbt.sources import duckdb_scan


def test_duckdb_scan_returns_lazy():
    """
    duckdb_scan should return a true LazyFrame, not an eager DataFrame
    wrapped in .lazy().

    Using .pl(lazy=True) ensures query pushdown works correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"

        # Create a test database with some data
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE events (id INT, name VARCHAR)")
        con.execute("INSERT INTO events VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        con.close()

        os.environ["TEST_DUCKDB_PATH"] = str(db_path)

        try:
            result = duckdb_scan("events", connection_key="TEST_DUCKDB_PATH")

            # Should be a LazyFrame
            assert isinstance(result, pl.LazyFrame), "duckdb_scan should return LazyFrame"

            # Verify data is correct when collected
            df = result.collect()
            assert len(df) == 3
            assert df["id"].to_list() == [1, 2, 3]
        finally:
            del os.environ["TEST_DUCKDB_PATH"]
