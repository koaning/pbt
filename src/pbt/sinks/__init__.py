"""PBT sinks for writing data to various storage backends."""

from pbt.sinks.base import DryRunPlan, Sink, WriteResult
from pbt.sinks.duckdb import DuckDBSink
from pbt.sinks.local import LocalSink
from pbt.sinks.s3 import S3Sink

__all__ = ["Sink", "WriteResult", "DryRunPlan", "LocalSink", "DuckDBSink", "S3Sink"]
