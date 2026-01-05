"""S3-compatible sink with Hive-style partitioning support."""

from typing import TYPE_CHECKING, Optional

import polars as pl

from .base import DryRunPlan, Sink, WriteResult

if TYPE_CHECKING:
    pass


class S3Sink(Sink):
    """S3-compatible sink with Hive-style partitioning.

    Writes tables as Parquet files to S3 or S3-compatible storage (MinIO, R2, etc.).
    Supports:
    - Single file writes (no partitioning)
    - Hive-style partitioned writes (e.g., date=2025-01-15/data.parquet)
    - Append and overwrite partition modes

    Args:
        bucket: S3 bucket name
        prefix: Key prefix (like a directory path), defaults to empty
        storage_options: Dict passed to Polars for S3 auth/config.
            Common keys: aws_access_key_id, aws_secret_access_key,
            endpoint_url (for S3-compatible services), region

    Example:
        sink = S3Sink(
            bucket="my-bucket",
            prefix="warehouse/",
            storage_options={
                "aws_access_key_id": "...",
                "aws_secret_access_key": "...",
            }
        )

        # Single file write
        sink.write(df, "my_table")
        # -> s3://my-bucket/warehouse/my_table.parquet

        # Partitioned write
        sink.write(df, "events", partition_by=["date"])
        # -> s3://my-bucket/warehouse/events/date=2025-01-15/data.parquet

        # S3-compatible (MinIO)
        sink = S3Sink(
            bucket="data",
            prefix="tables/",
            storage_options={"endpoint_url": "http://localhost:9000", ...}
        )
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        storage_options: Optional[dict] = None,
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3Sink. Install with: pip install boto3"
            )

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.storage_options = storage_options or {}

        # Build boto3 client config from storage_options
        client_kwargs = {}
        if "endpoint_url" in self.storage_options:
            client_kwargs["endpoint_url"] = self.storage_options["endpoint_url"]
        if "region" in self.storage_options:
            client_kwargs["region_name"] = self.storage_options["region"]
        if "aws_access_key_id" in self.storage_options:
            client_kwargs["aws_access_key_id"] = self.storage_options[
                "aws_access_key_id"
            ]
        if "aws_secret_access_key" in self.storage_options:
            client_kwargs["aws_secret_access_key"] = self.storage_options[
                "aws_secret_access_key"
            ]

        self._client = boto3.client("s3", **client_kwargs)

    def _s3_uri(self, *parts: str) -> str:
        """Build s3:// URI from parts."""
        path_parts = [self.prefix] if self.prefix else []
        path_parts.extend(parts)
        path = "/".join(p.strip("/") for p in path_parts if p)
        return f"s3://{self.bucket}/{path}"

    def _s3_key(self, *parts: str) -> str:
        """Build S3 key (without s3:// prefix) from parts."""
        path_parts = [self.prefix] if self.prefix else []
        path_parts.extend(parts)
        return "/".join(p.strip("/") for p in path_parts if p)

    def write(
        self,
        df: pl.DataFrame,
        name: str,
        partition_by: Optional[list[str]] = None,
        partition_mode: str = "append",
        dry_run: bool = False,
    ) -> WriteResult | DryRunPlan:
        """Write DataFrame to S3.

        Args:
            df: Data to write
            name: Table name (becomes key prefix)
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
        s3_uri = self._s3_uri(f"{name}.parquet")
        key = self._s3_key(f"{name}.parquet")

        exists = self._object_exists(key)

        if dry_run:
            return DryRunPlan(
                table_name=name,
                rows_to_write=len(df),
                partitions_affected=[],
                partition_operations={},
                destination=s3_uri,
            )

        operation = "overwrite" if exists else "create"
        df.write_parquet(s3_uri, storage_options=self.storage_options)

        return WriteResult(
            rows_written=len(df),
            partitions_affected=[],
            operation=operation,
            destination=s3_uri,
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

            key = self._s3_key(name, partition_path, "data.parquet")
            exists = self._object_exists(key)

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
                destination=self._s3_uri(name),
            )

        # Actual write
        total_rows = 0
        for pv in partition_values:
            partition_path_parts = [f"{k}={v}" for k, v in pv.items()]
            partition_path = "/".join(partition_path_parts)

            # Filter data for this partition
            filter_expr = pl.lit(True)
            for col, val in pv.items():
                filter_expr = filter_expr & (pl.col(col) == val)
            partition_df = df.filter(filter_expr)

            # Remove partition columns from data (they're encoded in path)
            data_cols = [c for c in partition_df.columns if c not in partition_by]
            partition_df = partition_df.select(data_cols)

            s3_uri = self._s3_uri(name, partition_path, "data.parquet")
            key = self._s3_key(name, partition_path, "data.parquet")

            if partition_mode == "overwrite" or not self._object_exists(key):
                partition_df.write_parquet(s3_uri, storage_options=self.storage_options)
            else:  # append to existing
                existing = pl.read_parquet(s3_uri, storage_options=self.storage_options)
                combined = pl.concat([existing, partition_df])
                combined.write_parquet(s3_uri, storage_options=self.storage_options)

            total_rows += len(partition_df)

        return WriteResult(
            rows_written=total_rows,
            partitions_affected=affected_partitions,
            operation=f"partitioned_{partition_mode}",
            destination=self._s3_uri(name),
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
        # Check for single file first
        single_file_key = self._s3_key(f"{name}.parquet")
        if self._object_exists(single_file_key):
            lf = pl.scan_parquet(
                self._s3_uri(f"{name}.parquet"),
                storage_options=self.storage_options,
            )
        else:
            # Assume partitioned - scan all parquet files under table prefix
            table_uri = self._s3_uri(name, "**/*.parquet")
            lf = pl.scan_parquet(
                table_uri,
                storage_options=self.storage_options,
                hive_partitioning=True,
            )

        # Apply partition filter if provided
        if partition_filter:
            for col, value in partition_filter.items():
                lf = lf.filter(pl.col(col) == value)

        return lf

    def exists(self, name: str) -> bool:
        """Check if table exists (single file or partitioned directory)."""
        # Check single file
        if self._object_exists(self._s3_key(f"{name}.parquet")):
            return True

        # Check for any objects under table prefix (partitioned)
        prefix = self._s3_key(name) + "/"
        response = self._client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        return response.get("KeyCount", 0) > 0

    def list_partitions(self, name: str) -> list[str]:
        """List Hive-style partition paths for a table."""
        prefix = self._s3_key(name) + "/"
        partitions = set()

        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract partition path (everything between table name and data.parquet)
                rel_key = key[len(prefix) :]
                if "/" in rel_key:
                    # partition path is everything except the filename
                    partition_path = "/".join(rel_key.split("/")[:-1])
                    if partition_path and "=" in partition_path:
                        partitions.add(partition_path)

        return sorted(partitions)

    def delete_partitions(self, name: str, partitions: list[str]) -> None:
        """Delete specific partitions from a table."""
        for partition in partitions:
            prefix = self._s3_key(name, partition) + "/"

            # List all objects in this partition
            paginator = self._client.get_paginator("list_objects_v2")
            objects_to_delete = []

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects_to_delete.append({"Key": obj["Key"]})

            # Batch delete (max 1000 per request)
            while objects_to_delete:
                batch = objects_to_delete[:1000]
                objects_to_delete = objects_to_delete[1000:]

                if batch:
                    self._client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": batch},
                    )

    def _object_exists(self, key: str) -> bool:
        """Check if a specific S3 object exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
