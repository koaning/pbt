"""PBT - Polars-based transformation tool inspired by DBT"""

from pbt.core import conf
from pbt.exceptions import PartitionSchemaError, PBTError
from pbt.extensions import setup_polars_extensions
from pbt.sinks import LocalSink
from pbt.sources import duckdb_scan

# Setup Polars extensions on import
setup_polars_extensions()

__all__ = [
    "conf",
    "LocalSink",
    "duckdb_scan",
    "PBTError",
    "PartitionSchemaError",
]
