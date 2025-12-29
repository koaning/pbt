"""DuckDB/MotherDuck source helper for PBT."""

import os
from typing import Optional

import polars as pl


def duckdb_scan(
    table: str,
    connection_key: str = "DUCKDB_PATH",
    query: Optional[str] = None,
) -> pl.LazyFrame:
    """Scan a DuckDB or MotherDuck table, returning a LazyFrame.

    This helper reads from DuckDB databases (local or MotherDuck) and returns
    a Polars LazyFrame for further processing. The connection string is read
    from an environment variable for security.

    Args:
        table: Name of the table to scan (ignored if query is provided)
        connection_key: Name of environment variable containing the connection string.
            For local DuckDB: path to the .duckdb file
            For MotherDuck: token starting with "md:" or "motherduck:"
        query: Optional custom SQL query. If provided, table is ignored.

    Returns:
        LazyFrame containing the table data

    Raises:
        ValueError: If the environment variable is not set
        ImportError: If duckdb is not installed

    Example:
        # Local DuckDB
        # export DUCKDB_PATH=/path/to/my.duckdb
        df = duckdb_scan("events")

        # MotherDuck
        # export MOTHERDUCK_TOKEN=md:your_token_here
        df = duckdb_scan("events", connection_key="MOTHERDUCK_TOKEN")

        # Custom query
        df = duckdb_scan("events", query="SELECT * FROM events WHERE year = 2025")
    """
    try:
        import duckdb
    except ImportError as e:
        raise ImportError(
            "duckdb is required for duckdb_scan(). "
            "Install it with: pip install duckdb"
        ) from e

    # Get connection string from environment
    connection_string = os.environ.get(connection_key)
    if connection_string is None:
        raise ValueError(
            f"Environment variable '{connection_key}' not set. "
            "Set it to a DuckDB file path or MotherDuck token.\n"
            "  For local DuckDB: export DUCKDB_PATH=/path/to/database.duckdb\n"
            "  For MotherDuck: export MOTHERDUCK_TOKEN=md:your_token_here"
        )

    # Build the query
    sql = query if query else f"SELECT * FROM {table}"

    # Connect and execute
    con = duckdb.connect(connection_string)
    try:
        result = con.execute(sql).pl()
    finally:
        con.close()

    # Return as lazy for consistency with PBT pipeline
    return result.lazy()
