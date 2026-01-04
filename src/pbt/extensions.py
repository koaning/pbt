"""Polars LazyFrame extensions for PBT functionality"""

from datetime import timedelta

import polars as pl

from pbt.utils import parse_timestamp_value


def incremental_filter(self, column: str):
    """Filter dataframe for incremental processing based on last run state

    Args:
        column: The timestamp/date column to filter on

    Returns:
        Filtered LazyFrame with only new records, or original if first run.
        If lookback is configured, includes records from (last_max_value - lookback).
    """
    # Get metadata injected by PBT
    meta = getattr(self, "_pbt_metadata", None)

    if meta is None:
        # No metadata means we're outside PBT execution context
        return self

    target_table = meta.get("target_table")
    schema_manager = meta.get("schema_manager")
    full_refresh = meta.get("full_refresh", False)
    rerun_range = meta.get("reprocess_range")
    lookback: timedelta | None = meta.get("lookback")

    if not target_table or not schema_manager:
        return self

    # Check rerun overrides first – these take precedence over incremental state
    if rerun_range:
        start_value, end_value = rerun_range
        expr = pl.col(column) >= pl.lit(parse_timestamp_value(start_value))
        if end_value:
            expr = expr & (pl.col(column) <= pl.lit(parse_timestamp_value(end_value)))
        return self.filter(expr)

    # Check if this is a full refresh or first run
    if full_refresh:
        return self

    state = schema_manager.get_state(target_table)
    last_max_value = state.get("last_max_value")

    if last_max_value is None:
        # First run - process everything
        return self

    # Parse the last max value
    filter_timestamp = parse_timestamp_value(last_max_value)

    # Apply lookback if configured
    if lookback is not None:
        filter_timestamp = filter_timestamp - lookback

    # Filter for records after the filter point
    # Use >= when lookback is set (to include the boundary), > otherwise
    if lookback is not None:
        return self.filter(pl.col(column) >= pl.lit(filter_timestamp))
    else:
        return self.filter(pl.col(column) > pl.lit(filter_timestamp))


def setup_polars_extensions():
    """Monkeypatch Polars LazyFrame with PBT methods"""
    # Add metadata attribute if it doesn't exist
    if not hasattr(pl.LazyFrame, "_pbt_metadata"):
        pl.LazyFrame._pbt_metadata = None

    # Add incremental_filter method
    pl.LazyFrame.incremental_filter = incremental_filter
