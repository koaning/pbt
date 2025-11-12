"""
Process user events using PBT (Polars-Based Transformations).

This example demonstrates:
- @app.source for raw data ingestion
- @app.model for intermediate transformations
- @app.table for materialized outputs
- Incremental processing with @app.incremental_table
"""

import polars as pl
from pathlib import Path
import sys

# Add parent directory to path to import pbt
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pbt import conf

# Initialize PBT app with example directory as root
app = conf(root=Path(__file__).parent)


@app.source
def raw_events():
    """Load raw events from CSV - always recomputed"""
    csv_path = Path(__file__).parent / "raw.csv"
    return pl.scan_csv(csv_path).with_columns(
        pl.col('timestamp').str.to_datetime()
    )


@app.model
def cleaned_events(raw_events):
    """Clean and validate events - lazy view, not materialized"""
    return raw_events.filter(
        pl.col('user').is_not_null() &
        pl.col('event').is_not_null() &
        pl.col('timestamp').is_not_null()
    )


@app.incremental_table(time_column="timestamp")
def events(cleaned_events):
    """
    Materialized events table with incremental processing.

    On subsequent runs, only processes events with timestamp > last max.
    """
    return (
        cleaned_events
        .select(['user', 'event', 'timestamp'])
    )


@app.table
def user_summary(cleaned_events):
    """Aggregated user statistics - full refresh each run"""
    return (
        cleaned_events
        .group_by('user')
        .agg([
            pl.len().alias('total_events'),
            pl.col('timestamp').min().alias('first_event'),
            pl.col('timestamp').max().alias('last_event')
        ])
        .with_columns(
            ((pl.col('last_event') - pl.col('first_event')).dt.total_seconds() / 60)
            .alias('session_duration_minutes')
        )
        .sort('user')
    )


@app.table
def daily_stats(cleaned_events):
    """Daily event statistics - full refresh each run"""
    return (
        cleaned_events
        .with_columns(pl.col('timestamp').dt.date().alias('date'))
        .group_by('date')
        .agg([
            pl.len().alias('total_events'),
            pl.col('user').n_unique().alias('unique_users'),
            pl.col('event').filter(pl.col('event') == 'purchase').len().alias('purchases')
        ])
        .sort('date')
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process user events with PBT")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--full-refresh", action="store_true", help="Force full refresh of all tables")
    args = parser.parse_args()

    # Run the PBT pipeline
    app.run(debug=args.debug, full_refresh=args.full_refresh)

    print("\n" + "=" * 60)
    print("DEMO: Model API")
    print("=" * 60)

    # Use .build() to read materialized table (eager)
    print("\nEvents table (using events.build()):")
    events_df = events.build()
    print(events_df.tail())

    # Use .build_lazy() to get lazy scan
    print("\nUser summary (using user_summary.build_lazy() for lazy scan):")
    user_summary_lf = user_summary.build_lazy()
    print(user_summary_lf.collect())
