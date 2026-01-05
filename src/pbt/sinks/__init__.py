"""PBT sinks for writing data to various storage backends."""

from pbt.sinks.base import DryRunPlan, Sink, WriteResult
from pbt.sinks.duckdb import DuckDBSink
from pbt.sinks.local import LocalSink

__all__ = ["Sink", "WriteResult", "DryRunPlan", "LocalSink", "DuckDBSink"]
