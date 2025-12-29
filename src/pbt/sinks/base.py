"""Base classes and interfaces for PBT sinks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import polars as pl


@dataclass
class WriteResult:
    """Result of a write operation."""

    rows_written: int
    partitions_affected: list[str] = field(default_factory=list)
    operation: str = "write"  # "create", "append", "overwrite"
    destination: str = ""


@dataclass
class DryRunPlan:
    """Plan showing what would happen without executing."""

    table_name: str
    rows_to_write: int
    partitions_affected: list[str] = field(default_factory=list)
    partition_operations: dict[str, str] = field(default_factory=dict)  # partition -> operation
    destination: str = ""


class Sink(ABC):
    """Abstract base class for PBT sinks.

    A sink handles writing data to storage, with optional partitioning support.
    """

    @abstractmethod
    def write(
        self,
        df: pl.DataFrame,
        name: str,
        partition_by: Optional[list[str]] = None,
        partition_mode: str = "append",
        dry_run: bool = False,
    ) -> WriteResult | DryRunPlan:
        """Write DataFrame to storage.

        Args:
            df: Data to write
            name: Table name
            partition_by: Columns to partition by (Hive-style)
            partition_mode: "append" or "overwrite" for partition handling
            dry_run: If True, return plan without executing

        Returns:
            WriteResult on actual write, DryRunPlan on dry_run=True
        """
        pass

    @abstractmethod
    def read(self, name: str, partition_filter: Optional[dict] = None) -> pl.LazyFrame:
        """Read table as LazyFrame, optionally filtering to specific partitions.

        Args:
            name: Table name
            partition_filter: Dict of column -> value to filter partitions

        Returns:
            LazyFrame of the table data
        """
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if table exists."""
        pass

    @abstractmethod
    def list_partitions(self, name: str) -> list[str]:
        """List existing partition paths for a table."""
        pass

    @abstractmethod
    def delete_partitions(self, name: str, partitions: list[str]) -> None:
        """Delete specific partitions from a table."""
        pass
