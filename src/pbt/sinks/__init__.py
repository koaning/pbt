"""PBT sinks for writing data to various storage backends."""

from pbt.sinks.base import DryRunPlan, Sink, WriteResult
from pbt.sinks.local import LocalSink
from pbt.sinks.s3 import S3Sink

__all__ = ["Sink", "WriteResult", "DryRunPlan", "LocalSink", "S3Sink"]
