"""Custom exceptions for PBT."""


class PBTError(Exception):
    """Base exception for PBT."""

    pass


class PartitionSchemaError(PBTError):
    """Raised when partition configuration has changed between runs.

    This error occurs when:
    - partition_by columns have changed
    - partition_mode has changed

    Resolution:
    - Run with full_refresh=True to rewrite the table
    - Or explicitly migrate the data
    """

    pass


class CircularDependencyError(PBTError):
    """Raised when a circular dependency is detected in the model graph.

    This error occurs when models depend on each other in a cycle,
    e.g., model A depends on B, and B depends on A.

    Resolution:
    - Review the dependency graph and break the cycle
    - Check function parameter names match the correct upstream models
    """

    pass


class IncrementalConfigError(PBTError):
    """Raised when incremental table configuration is invalid.

    This error occurs when:
    - time_column doesn't exist in the DataFrame
    - time_column is not a datetime-like type
    - incremental_table is missing required date partition

    Resolution:
    - Ensure time_column exists and is a Datetime or Date type
    - Add a date partition derived from time_column
    """

    pass
