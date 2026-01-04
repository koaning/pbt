"""PBT sinks for writing data to various storage backends."""

from pbt.sinks.base import DryRunPlan, Sink, WriteResult
from pbt.sinks.local import LocalSink

__all__ = ["Sink", "WriteResult", "DryRunPlan", "LocalSink"]
