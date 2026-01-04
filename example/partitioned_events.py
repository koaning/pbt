"""
Process user events with partitioning using PBT.

This example demonstrates:
- Custom sinks with LocalSink
- Hive-style partitioned tables
- Partition modes (overwrite vs append)
- Dry-run to preview changes
- Schema change detection
"""

import sys
from pathlib import Path

import polars as pl

# Add parent directory to path to import pbt
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pbt import LocalSink, conf

# Initialize PBT app
app = conf(root=Path(__file__).parent)

# Create a custom sink for partitioned output
partitioned_sink = LocalSink(path=Path(__file__).parent / "partitioned_output")


@app.source
def raw_events():
    """Load raw events from CSV."""
    csv_path = Path(__file__).parent / "raw.csv"
    return pl.scan_csv(csv_path).with_columns(pl.col("timestamp").str.to_datetime())


@app.model
def events_with_date(raw_events):
    """Add date column for partitioning."""
    return raw_events.with_columns(pl.col("timestamp").dt.date().alias("date"))


@app.table(
    sink=partitioned_sink,
    partition_by="date",
    partition_mode="overwrite",
)
def events_by_date(events_with_date):
    """
    Events partitioned by date.

    Uses partition_mode="overwrite" - each partition is fully rewritten
    when new data arrives for that date. This ensures correctness when
    data may be updated or corrected.

    Output structure:
        partitioned_output/events_by_date/
            date=2025-01-15/data.parquet
            date=2025-01-16/data.parquet
            ...
    """
    return events_with_date.select(["user", "event", "timestamp", "date"])


@app.incremental_table(
    time_column="timestamp",
    partition_by="date",
    partition_mode="append",
    sink=partitioned_sink,
)
def events_incremental(events_with_date):
    """
    Incremental events partitioned by date.

    Uses partition_mode="append" - new records are appended to existing
    partitions. Only use this when you're certain data won't be updated
    (e.g., append-only event logs).

    Combined with incremental_table, this only processes new records
    (timestamp > last_max_value) and appends them to the appropriate
    date partition.
    """
    return events_with_date.select(["user", "event", "timestamp", "date"])


@app.table(
    sink=partitioned_sink,
    partition_by=["user", "date"],
)
def events_by_user_date(events_with_date):
    """
    Events partitioned by both user and date.

    Multi-level partitioning creates nested directories:
        partitioned_output/events_by_user_date/
            user=alice/
                date=2025-01-15/data.parquet
                date=2025-01-16/data.parquet
            user=bob/
                date=2025-01-15/data.parquet
                ...
    """
    return events_with_date.select(["user", "event", "timestamp", "date"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process events with partitioning")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--full-refresh", action="store_true", help="Force full refresh"
    )
    parser.add_argument(
        "--target", type=str, help="Only run this table and its dependencies"
    )
    args = parser.parse_args()

    # Run the pipeline
    result = app.run(
        target=args.target,
        debug=args.debug,
        dry_run=args.dry_run,
        full_refresh=args.full_refresh,
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE - No files were written")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("PARTITIONS CREATED")
        print("=" * 60)

        for table_name in ["events_by_date", "events_incremental", "events_by_user_date"]:
            if args.target and args.target != table_name:
                continue
            partitions = partitioned_sink.list_partitions(table_name)
            if partitions:
                print(f"\n{table_name}/ ({len(partitions)} partitions)")
                for p in partitions[:5]:
                    print(f"  - {p}")
                if len(partitions) > 5:
                    print(f"  ... and {len(partitions) - 5} more")

        # Demo: Read back partitioned data
        print("\n" + "=" * 60)
        print("READING PARTITIONED DATA")
        print("=" * 60)

        print("\nAll events_by_date (Polars reconstructs 'date' from path):")
        df = partitioned_sink.read("events_by_date").collect()
        print(df.head(5))

        print("\nFilter to specific date (partition pruning):")
        from datetime import date
        df_filtered = partitioned_sink.read(
            "events_by_date", partition_filter={"date": date(2025, 1, 15)}
        ).collect()
        print(f"Rows for 2025-01-15: {len(df_filtered)}")
