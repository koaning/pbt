"""
Process user events using PBT (Polars-Based Transformations).

This example demonstrates:
- @app.source for raw data ingestion
- @app.model for intermediate transformations
- @app.table for materialized outputs
- Incremental processing with .incremental_filter()
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


@app.table(incremental=True, time_column="timestamp")
def events(cleaned_events):
    """
    Materialized events table with incremental processing.

    On subsequent runs, only processes events with timestamp > last max.
    """
    return (
        cleaned_events
        .incremental_filter("timestamp")
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


def print_table(title: str, table_name: str, output_dir: Path, n_rows: int = 10):
    """Load and print a table if it exists"""
    path = output_dir / f"{table_name}.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(df.head(n_rows))
    return df


def print_analysis():
    """Print analysis from materialized tables"""
    output_dir = Path(__file__).parent / "output"

    print("\n" + "=" * 60)
    print("ANALYSIS FROM MATERIALIZED TABLES")
    print("=" * 60)

    # Print all tables
    events_df = print_table("EVENTS", "events", output_dir)
    print_table("USER SUMMARY", "user_summary", output_dir)
    print_table("DAILY STATISTICS", "daily_stats", output_dir)

    # Conversion funnel from events
    if events_df is not None and len(events_df) > 0:
        total_users = events_df.select(pl.col('user').n_unique()).item()

        print("\n" + "-" * 60)
        print("CONVERSION FUNNEL")
        print("-" * 60)
        print(f"total_users: {total_users}")

        for event, metric_name in [
            ('login', 'login_rate'),
            ('page_view', 'page_view_rate'),
            ('click_button', 'click_rate'),
            ('purchase', 'conversion_rate')
        ]:
            count = events_df.filter(pl.col('event') == event).select(pl.col('user').n_unique()).item()
            print(f"{metric_name}: {count / total_users:.2%}")


if __name__ == "__main__":
    print("=" * 60)
    print("PBT EVENT PROCESSING PIPELINE")
    print("=" * 60)

    # Run the PBT pipeline
    app.run()

    # Print analysis
    print_analysis()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
