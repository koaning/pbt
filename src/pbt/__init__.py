"""PBT - Polars-based transformation tool inspired by DBT"""

from pbt.core import conf
from pbt.exceptions import (
    CircularDependencyError,
    IncrementalConfigError,
    PartitionSchemaError,
    PBTError,
)
from pbt.extensions import setup_polars_extensions
from pbt.sinks import LocalSink
from pbt.sources import duckdb_scan
from pbt.meta import ColumnMeta, TableMeta, SchemaChange, SchemaChangeError

# Setup Polars extensions on import
setup_polars_extensions()

__all__ = [
    "conf",
    "LocalSink",
    "duckdb_scan",
    "PBTError",
    "CircularDependencyError",
    "IncrementalConfigError",
    "PartitionSchemaError",
    "ColumnMeta",
    "TableMeta",
    "SchemaChange",
    "SchemaChangeError",
]
