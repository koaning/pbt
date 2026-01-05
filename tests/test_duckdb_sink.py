"""DuckDB-specific sink tests."""

import datetime
from pathlib import Path

import polars as pl
import pytest

from pbt.sinks import DuckDBSink


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Sample DataFrame for testing."""
    return pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestDuckDBConnection:
    """Test DuckDB connection handling."""

    def test_in_memory_database(self, sample_df: pl.DataFrame):
        """In-memory database works correctly."""
        sink = DuckDBSink(path=":memory:")
        sink.write(sample_df, "test")

        read_df = sink.read("test").collect()
        assert len(read_df) == 3
        sink.close()

    def test_file_database(self, tmp_path: Path, sample_df: pl.DataFrame):
        """File-based database persists data."""
        db_path = str(tmp_path / "test.db")

        # Write with one connection
        sink1 = DuckDBSink(path=db_path)
        sink1.write(sample_df, "test")
        sink1.close()

        # Read with new connection
        sink2 = DuckDBSink(path=db_path)
        assert sink2.exists("test")
        read_df = sink2.read("test").collect()
        assert len(read_df) == 3
        sink2.close()

    def test_close_and_reopen(self, tmp_path: Path, sample_df: pl.DataFrame):
        """Connection can be closed and reopened."""
        db_path = str(tmp_path / "test.db")
        sink = DuckDBSink(path=db_path)

        sink.write(sample_df, "test")
        sink.close()

        # Should reconnect automatically
        assert sink.exists("test")
        sink.close()


class TestMotherDuckConnection:
    """Test MotherDuck connection string handling.

    These tests verify the connection string parsing without actually
    connecting to MotherDuck (which requires authentication).
    """

    def test_motherduck_path_format(self):
        """MotherDuck path format is accepted (md:database)."""
        # This test just verifies the path is stored correctly
        # Actual connection would require MOTHERDUCK_TOKEN
        sink = DuckDBSink(path="md:my_database")
        assert sink.path == "md:my_database"
        # Don't try to connect - would fail without token


class TestMetadataTable:
    """Test partition metadata handling."""

    def test_metadata_table_created(self, tmp_path: Path):
        """Metadata table is created on first connection."""
        sink = DuckDBSink(path=str(tmp_path / "test.db"))
        conn = sink._get_conn()

        # Check metadata table exists
        result = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [sink._METADATA_TABLE],
        ).fetchone()
        assert result is not None
        sink.close()

    def test_partition_columns_stored(self, tmp_path: Path):
        """Partition columns are stored in metadata."""
        sink = DuckDBSink(path=str(tmp_path / "test.db"))

        df = pl.DataFrame(
            {
                "id": [1, 2],
                "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)],
            }
        )
        sink.write(df, "events", partition_by=["date"])

        # Check metadata was stored
        partition_cols = sink._get_partition_columns("events")
        assert partition_cols == ["date"]
        sink.close()

    def test_partition_columns_persisted(self, tmp_path: Path):
        """Partition columns persist across connections."""
        db_path = str(tmp_path / "test.db")

        df = pl.DataFrame(
            {
                "id": [1, 2],
                "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)],
            }
        )

        # Write with one connection
        sink1 = DuckDBSink(path=db_path)
        sink1.write(df, "events", partition_by=["date"])
        sink1.close()

        # Read metadata with new connection
        sink2 = DuckDBSink(path=db_path)
        partition_cols = sink2._get_partition_columns("events")
        assert partition_cols == ["date"]
        sink2.close()


class TestMultiColumnPartitions:
    """Test partitioning by multiple columns."""

    def test_two_column_partition(self, tmp_path: Path):
        """Partitioning by two columns works correctly."""
        sink = DuckDBSink(path=str(tmp_path / "test.db"))

        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "year": [2024, 2024, 2025, 2025],
                "month": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        sink.write(df, "events", partition_by=["year", "month"])

        partitions = sink.list_partitions("events")
        assert len(partitions) == 4
        assert "year=2024/month=1" in partitions
        assert "year=2024/month=2" in partitions
        assert "year=2025/month=1" in partitions
        assert "year=2025/month=2" in partitions
        sink.close()

    def test_delete_multi_column_partition(self, tmp_path: Path):
        """Deleting multi-column partitions works correctly."""
        sink = DuckDBSink(path=str(tmp_path / "test.db"))

        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "year": [2024, 2024, 2025, 2025],
                "month": [1, 2, 1, 2],
            }
        )
        sink.write(df, "events", partition_by=["year", "month"])

        sink.delete_partitions("events", ["year=2024/month=1"])

        partitions = sink.list_partitions("events")
        assert len(partitions) == 3
        assert "year=2024/month=1" not in partitions
        sink.close()
