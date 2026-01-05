"""Local filesystem sink with Hive-style partitioning support."""

import shutil
from pathlib import Path
from typing import Optional

import polars as pl

from .base import DryRunPlan, Sink, WriteResult


class LocalSink(Sink):
    """Local filesystem sink with Hive-style partitioning.

    Writes tables as Parquet files to a local directory. Supports:
    - Single file writes (no partitioning)
    - Hive-style partitioned writes (e.g., date=2025-01-15/data.parquet)
    - Append and overwrite partition modes

    Args:
        path: Base directory for output files

    Example:
        sink = LocalSink(path="./output")

        # Single file write
        sink.write(df, "my_table")
        # -> ./output/my_table.parquet

        # Partitioned write
        sink.write(df, "events", partition_by=["date"])
        # -> ./output/events/date=2025-01-15/data.parquet
    """

    def __init__(self, path: str | Path = "./output"):
        self.base_path = Path(path)

    def write(
        self,
        df: pl.DataFrame,
        name: str,
        partition_by: Optional[list[str]] = None,
        partition_mode: str = "append",
        dry_run: bool = False,
    ) -> WriteResult | DryRunPlan:
        """Write DataFrame to local filesystem.

        Args:
            df: Data to write
            name: Table name (becomes filename or directory name)
            partition_by: Columns to partition by (Hive-style)
            partition_mode: "append" or "overwrite" for partition handling
            dry_run: If True, return plan without executing

        Returns:
            WriteResult on actual write, DryRunPlan on dry_run=True
        """
        if partition_by:
            return self._write_partitioned(
                df, name, partition_by, partition_mode, dry_run
            )
        else:
            return self._write_single(df, name, dry_run)

    def _write_single(
        self, df: pl.DataFrame, name: str, dry_run: bool
    ) -> WriteResult | DryRunPlan:
        """Write as single parquet file."""
        output_path = self.base_path / f"{name}.parquet"

        if dry_run:
            operation = "overwrite" if output_path.exists() else "create"
            return DryRunPlan(
                table_name=name,
                rows_to_write=len(df),
                partitions_affected=[],
                partition_operations={},
                destination=str(output_path),
            )

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        operation = "overwrite" if output_path.exists() else "create"
        df.write_parquet(output_path)

        return WriteResult(
            rows_written=len(df),
            partitions_affected=[],
            operation=operation,
            destination=str(output_path),
        )

    def _write_partitioned(
        self,
        df: pl.DataFrame,
        name: str,
        partition_by: list[str],
        partition_mode: str,
        dry_run: bool,
    ) -> WriteResult | DryRunPlan:
        """Write with Hive-style partitioning."""
        table_path = self.base_path / name

        # Validate partition columns exist
        missing_cols = set(partition_by) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Partition columns not in DataFrame: {missing_cols}")

        # Determine affected partitions from data
        partition_values = df.select(partition_by).unique().to_dicts()

        affected_partitions = []
        partition_ops = {}

        for pv in partition_values:
            # Build Hive-style path: col1=val1/col2=val2/
            partition_path_parts = [f"{k}={v}" for k, v in pv.items()]
            partition_path = "/".join(partition_path_parts)
            affected_partitions.append(partition_path)

            full_partition_path = table_path / partition_path
            exists = full_partition_path.exists()

            if partition_mode == "overwrite":
                partition_ops[partition_path] = "overwrite" if exists else "create"
            else:  # append
                partition_ops[partition_path] = "append" if exists else "create"

        if dry_run:
            return DryRunPlan(
                table_name=name,
                rows_to_write=len(df),
                partitions_affected=affected_partitions,
                partition_operations=partition_ops,
                destination=str(table_path),
            )

        # Actual write
        total_rows = 0
        for pv in partition_values:
            partition_path_parts = [f"{k}={v}" for k, v in pv.items()]
            partition_path = "/".join(partition_path_parts)
            full_partition_path = table_path / partition_path

            # Filter data for this partition
            filter_expr = pl.lit(True)
            for col, val in pv.items():
                filter_expr = filter_expr & (pl.col(col) == val)
            partition_df = df.filter(filter_expr)

            # Remove partition columns from data (they're encoded in path)
            data_cols = [c for c in partition_df.columns if c not in partition_by]
            partition_df = partition_df.select(data_cols)

            full_partition_path.mkdir(parents=True, exist_ok=True)
            data_file = full_partition_path / "data.parquet"

            if partition_mode == "overwrite" or not data_file.exists():
                partition_df.write_parquet(data_file)
            else:  # append to existing
                existing = pl.read_parquet(data_file)
                combined = pl.concat([existing, partition_df])
                combined.write_parquet(data_file)

            total_rows += len(partition_df)

        return WriteResult(
            rows_written=total_rows,
            partitions_affected=affected_partitions,
            operation=f"partitioned_{partition_mode}",
            destination=str(table_path),
        )

    def read(self, name: str, partition_filter: Optional[dict] = None) -> pl.LazyFrame:
        """Read table as LazyFrame.

        Supports both single file and partitioned tables.
        For partitioned tables, uses Polars' hive_partitioning to reconstruct
        partition columns.

        Args:
            name: Table name
            partition_filter: Dict of column -> value to filter partitions

        Returns:
            LazyFrame of the table data
        """
        table_path = self.base_path / name

        # Check for partitioned directory
        if table_path.is_dir():
            # Scan all parquet files in partition structure
            lf = pl.scan_parquet(table_path / "**/*.parquet", hive_partitioning=True)
        else:
            # Single file
            parquet_path = self.base_path / f"{name}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Table '{name}' not found at {parquet_path}")
            lf = pl.scan_parquet(parquet_path)

        # Apply partition filter if provided
        if partition_filter:
            for col, value in partition_filter.items():
                lf = lf.filter(pl.col(col) == value)

        return lf

    def exists(self, name: str) -> bool:
        """Check if table exists (file or directory)."""
        return (self.base_path / f"{name}.parquet").exists() or (
            self.base_path / name
        ).is_dir()

    def list_partitions(self, name: str) -> list[str]:
        """List Hive-style partition paths for a table."""
        table_path = self.base_path / name
        if not table_path.is_dir():
            return []

        partitions = set()
        for parquet_file in table_path.rglob("*.parquet"):
            rel_path = parquet_file.relative_to(table_path).parent
            if str(rel_path) != ".":
                partitions.add(str(rel_path))

        return sorted(partitions)

    def delete_partitions(self, name: str, partitions: list[str]) -> None:
        """Delete specific partition directories."""
        table_path = self.base_path / name
        for partition in partitions:
            partition_path = table_path / partition
            if partition_path.exists():
                shutil.rmtree(partition_path)
