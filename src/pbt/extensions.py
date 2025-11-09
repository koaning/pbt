"""Polars LazyFrame extensions for PBT functionality"""

import polars as pl


def incremental_filter(self, column: str):
    """Filter dataframe for incremental processing based on last run state

    Args:
        column: The timestamp/date column to filter on

    Returns:
        Filtered LazyFrame with only new records, or original if first run
    """
    # Get metadata injected by PBT
    meta = getattr(self, '_pbt_metadata', None)

    if meta is None:
        # No metadata means we're outside PBT execution context
        return self

    target_table = meta.get("target_table")
    state_manager = meta.get("state_manager")
    full_refresh = meta.get("full_refresh", False)

    if not target_table or not state_manager:
        return self

    # Check if this is a full refresh or first run
    if full_refresh:
        return self

    state = state_manager.get_state(target_table)
    last_max_value = state.get("last_max_value")

    if last_max_value is None:
        # First run - process everything
        return self

    # Filter for records after last max value
    # Parse the string value back to appropriate type using pl.lit
    # pl.lit will infer the type and convert appropriately
    try:
        # Try parsing as datetime first
        from datetime import datetime
        parsed_value = datetime.fromisoformat(last_max_value.replace(" ", "T"))
        filter_value = pl.lit(parsed_value)
    except (ValueError, AttributeError):
        # Fall back to string comparison or numeric
        filter_value = pl.lit(last_max_value)

    return self.filter(pl.col(column) > filter_value)


def setup_polars_extensions():
    """Monkeypatch Polars LazyFrame with PBT methods"""
    # Add metadata attribute if it doesn't exist
    if not hasattr(pl.LazyFrame, '_pbt_metadata'):
        pl.LazyFrame._pbt_metadata = None

    # Add incremental_filter method
    pl.LazyFrame.incremental_filter = incremental_filter
