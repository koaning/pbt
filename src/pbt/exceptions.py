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
