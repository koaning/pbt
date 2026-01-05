"""DuckDB sink for writing data to DuckDB databases."""

from typing import Optional

import duckdb
import polars as pl

from .base import DryRunPlan, Sink, WriteResult


class DuckDBSink(Sink):
    """DuckDB sink for writing tables to a DuckDB database.

    Supports local DuckDB files, in-memory databases, and MotherDuck cloud.
    Partitions are stored as table columns (not separate files like LocalSink).

    Args:
        path: Database path. Options:
            - ":memory:" for in-memory database
            - "path/to/file.db" for local file
            - "md:database" for MotherDuck

    Example:
        sink = DuckDBSink(path="./output.db")

        # Simple write
        sink.write(df, "my_table")

        # Partitioned write
        sink.write(df, "events", partition_by=["date"])
    """

    # Metadata table to track partition columns per table
    _METADATA_TABLE = "_pbt_metadata"

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.path)
            self._ensure_metadata_table()
        return self._conn

    def _ensure_metadata_table(self) -> None:
        """Create metadata table if it doesn't exist."""
        conn = self._conn
        if conn is None:
            return
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._METADATA_TABLE} (
                table_name VARCHAR PRIMARY KEY,
                partition_columns VARCHAR[]
            )
        """)

    def _get_partition_columns(self, name: str) -> list[str]:
        """Get partition columns for a table from metadata."""
        conn = self._get_conn()
        result = conn.execute(
            f"SELECT partition_columns FROM {self._METADATA_TABLE} WHERE table_name = ?",
            [name],
        ).fetchone()
        if result and result[0]:
            return list(result[0])
        return []

    def _set_partition_columns(self, name: str, partition_by: list[str]) -> None:
        """Store partition columns for a table in metadata."""
        conn = self._get_conn()
        conn.execute(
            f"""
            INSERT INTO {self._METADATA_TABLE} (table_name, partition_columns)
            VALUES (?, ?)
            ON CONFLICT (table_name) DO UPDATE SET partition_columns = EXCLUDED.partition_columns
            """,
            [name, partition_by],
        )

    def write(
        self,
        df: pl.DataFrame,
        name: str,
        partition_by: Optional[list[str]] = None,
        partition_mode: str = "append",
        dry_run: bool = False,
    ) -> WriteResult | DryRunPlan:
        """Write DataFrame to DuckDB table.

        Args:
            df: Data to write
            name: Table name
            partition_by: Columns to treat as partitions (stored as regular columns)
            partition_mode: "append" or "overwrite" for partition handling
            dry_run: If True, return plan without executing

        Returns:
            WriteResult on actual write, DryRunPlan on dry_run=True
        """
        conn = self._get_conn()
        table_exists = self.exists(name)

        # Validate partition columns exist in DataFrame
        if partition_by:
            missing_cols = set(partition_by) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Partition columns not in DataFrame: {missing_cols}")

        # Determine affected partitions
        affected_partitions: list[str] = []
        partition_ops: dict[str, str] = {}

        if partition_by:
            partition_values = df.select(partition_by).unique().to_dicts()
            for pv in partition_values:
                partition_path = "/".join(f"{k}={v}" for k, v in pv.items())
                affected_partitions.append(partition_path)

                # Check if partition exists
                if table_exists:
                    filter_conditions = " AND ".join(
                        f"{k} = {_sql_value(v)}" for k, v in pv.items()
                    )
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {name} WHERE {filter_conditions}"
                    ).fetchone()
                    exists = count is not None and count[0] > 0
                else:
                    exists = False

                if partition_mode == "overwrite":
                    partition_ops[partition_path] = "overwrite" if exists else "create"
                else:
                    partition_ops[partition_path] = "append" if exists else "create"

        if dry_run:
            return DryRunPlan(
                table_name=name,
                rows_to_write=len(df),
                partitions_affected=affected_partitions,
                partition_operations=partition_ops,
                destination=f"duckdb://{self.path}/{name}",
            )

        # Actual write
        if not table_exists:
            # Create table from DataFrame
            conn.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
            operation = "create"
        else:
            if partition_by and partition_mode == "overwrite":
                # Delete existing data for affected partitions
                partition_values = df.select(partition_by).unique().to_dicts()
                for pv in partition_values:
                    filter_conditions = " AND ".join(
                        f"{k} = {_sql_value(v)}" for k, v in pv.items()
                    )
                    conn.execute(f"DELETE FROM {name} WHERE {filter_conditions}")

            # Insert new data
            conn.execute(f"INSERT INTO {name} SELECT * FROM df")
            operation = f"partitioned_{partition_mode}" if partition_by else "append"

        # Store partition metadata
        if partition_by:
            self._set_partition_columns(name, partition_by)

        return WriteResult(
            rows_written=len(df),
            partitions_affected=affected_partitions,
            operation=operation,
            destination=f"duckdb://{self.path}/{name}",
        )

    def read(self, name: str, partition_filter: Optional[dict] = None) -> pl.LazyFrame:
        """Read table as LazyFrame.

        Args:
            name: Table name
            partition_filter: Dict of column -> value to filter partitions

        Returns:
            LazyFrame of the table data
        """
        conn = self._get_conn()

        if not self.exists(name):
            raise ValueError(f"Table '{name}' not found in database")

        # Build query with optional partition filter
        query = f"SELECT * FROM {name}"
        if partition_filter:
            conditions = " AND ".join(
                f"{k} = {_sql_value(v)}" for k, v in partition_filter.items()
            )
            query += f" WHERE {conditions}"

        # Execute query and convert to Polars LazyFrame
        result = conn.execute(query).pl()
        return result.lazy()

    def exists(self, name: str) -> bool:
        """Check if table exists in database."""
        conn = self._get_conn()
        result = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        return result is not None

    def list_partitions(self, name: str) -> list[str]:
        """List partition values as Hive-style paths.

        Returns partition values formatted as "col1=val1/col2=val2".
        """
        conn = self._get_conn()

        if not self.exists(name):
            return []

        partition_cols = self._get_partition_columns(name)
        if not partition_cols:
            return []

        # Query distinct partition values
        cols_str = ", ".join(partition_cols)
        result = conn.execute(f"SELECT DISTINCT {cols_str} FROM {name}").fetchall()

        # Format as Hive-style paths
        partitions = []
        for row in result:
            parts = [f"{col}={val}" for col, val in zip(partition_cols, row)]
            partitions.append("/".join(parts))

        return sorted(partitions)

    def delete_partitions(self, name: str, partitions: list[str]) -> None:
        """Delete rows matching partition specifications.

        Args:
            name: Table name
            partitions: List of Hive-style partition paths (e.g., "date=2025-01-15")
        """
        conn = self._get_conn()

        if not self.exists(name):
            return

        for partition in partitions:
            # Parse Hive-style path into column-value pairs
            conditions = []
            for part in partition.split("/"):
                if "=" in part:
                    col, val = part.split("=", 1)
                    conditions.append(f"{col} = {_sql_value(val)}")

            if conditions:
                where_clause = " AND ".join(conditions)
                conn.execute(f"DELETE FROM {name} WHERE {where_clause}")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _sql_value(value) -> str:
    """Convert a Python value to SQL literal."""
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        # Escape single quotes
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        # For dates and other types, convert to string
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
