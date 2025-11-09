"""Example PBT pipeline with incremental processing"""

import polars as pl
from datetime import datetime, timedelta
from pathlib import Path

from pbt import conf

# Initialize app
app = conf(root="./data", env="dev")


@app.source
def raw_logs():
    """Load raw log data from parquet files"""
    # For demo: create sample data if it doesn't exist
    log_path = Path("data/logs/events.parquet")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        # Create sample data
        sample_data = pl.DataFrame({
            "user_id": [1, 2, 3, 1, 2] * 20,
            "event": ["login", "click", "purchase", "logout", "click"] * 20,
            "timestamp": [
                datetime(2025, 11, 1) + timedelta(hours=i)
                for i in range(100)
            ],
            "value": list(range(100))
        })
        sample_data.write_parquet(log_path)

    return pl.scan_parquet(log_path)


@app.model
def cleaned_logs(raw_logs):
    """Clean and filter logs - just a view, not materialized"""
    return (
        raw_logs
        .filter(pl.col("event").is_not_null())
        .with_columns([
            pl.col("timestamp").alias("event_time"),
            pl.col("user_id").cast(pl.Int64)
        ])
    )


@app.table(incremental=True, time_column="event_time")
def user_events(cleaned_logs):
    """Incremental table tracking all user events"""
    return (
        cleaned_logs
        .incremental_filter("event_time")  # Only process new events
        .select([
            "user_id",
            "event",
            "event_time",
            "value"
        ])
    )


@app.table
def daily_summary(cleaned_logs):
    """Daily aggregation - full refresh each time"""
    return (
        cleaned_logs
        .with_columns(pl.col("event_time").dt.date().alias("date"))
        .group_by("date")
        .agg([
            pl.count().alias("event_count"),
            pl.col("user_id").n_unique().alias("unique_users"),
            pl.col("value").sum().alias("total_value")
        ])
        .sort("date")
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Running PBT pipeline...")
    print("=" * 50)

    # Run all models
    app.run()

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)

    # Show outputs
    print("\nGenerated tables:")
    output_dir = Path("data/output")
    for f in sorted(output_dir.glob("*.parquet")):
        df = pl.read_parquet(f)
        print(f"\n{f.name} ({len(df)} rows):")
        print(df.head())
