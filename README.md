# PBT - Polars-Based Transformations

A DBT-like data transformation tool built on Polars/Narwhals with lazy evaluation and incremental processing.

## Features

- **Decorator-based model definition** - `@app.source`, `@app.model`, `@app.table`
- **Automatic dependency resolution** - DAG built from function parameters
- **Incremental processing** - Track state and process only new records
- **Lazy evaluation** - All models use Polars LazyFrames for efficiency
- **Simple materialization** - Tables write to parquet automatically

## Quick Start

```python
from pbt import conf
import polars as pl

app = conf(root="./data")

@app.source
def raw_logs():
    """Sources are always recomputed"""
    return pl.scan_parquet("logs/*.parquet")

@app.model
def cleaned_logs(raw_logs):
    """Models are views - lazy evaluation, not materialized"""
    return raw_logs.filter(pl.col("event").is_not_null())

@app.table(incremental=True, time_column="timestamp")
def user_events(cleaned_logs):
    """Tables are materialized to parquet"""
    return (
        cleaned_logs
        .incremental_filter("timestamp")  # Only new records
        .select(["user_id", "event", "timestamp"])
    )

@app.table
def daily_summary(cleaned_logs):
    """Non-incremental table - full refresh"""
    return (
        cleaned_logs
        .group_by(pl.col("timestamp").dt.date())
        .agg(pl.len().alias("count"))
    )

# Run the pipeline
app.run()
```

## Model Types

### `@app.source`
- Always recomputed on each run
- Use for reading raw data files
- Should return a LazyFrame

### `@app.model`
- Lazy views - not materialized
- Useful for intermediate transformations
- Passed as parameters to downstream models

### `@app.table`
- Materialized to parquet files in `{root}/output/`
- Supports incremental processing
- Supports full refresh mode

## Incremental Processing

Incremental tables track the maximum value of a time column and only process new records:

```python
@app.table(incremental=True, time_column="created_at")
def events(raw_data):
    return (
        raw_data
        .incremental_filter("created_at")  # Filters where created_at > last_max
        .select(...)
    )
```

### How it works:

1. **First run**: Processes all data, saves max value of `time_column`
2. **Subsequent runs**:
   - `.incremental_filter()` automatically adds `WHERE col > last_max_value`
   - New records are appended to existing parquet file
   - State is updated with new max value

State is stored in `{root}/.pbt/state.json`:

```json
{
  "events": {
    "last_max_value": "2025-11-05 23:00:00",
    "last_run": "..."
  }
}
```

## Dependency Resolution

PBT builds a DAG by inspecting function parameters:

```python
@app.source
def source_a():
    return pl.scan_parquet("a.parquet")

@app.model
def model_b(source_a):  # Depends on source_a
    return source_a.filter(...)

@app.table
def table_c(model_b, source_a):  # Depends on both
    return model_b.join(source_a, ...)
```

Execution order is automatically determined via topological sort.

## Implementation Details

### Metadata Injection

PBT monkeypatches `pl.LazyFrame` to add `._pbt_metadata`:

```python
# Automatically injected during execution
df._pbt_metadata = {
    "target_table": "user_events",
    "state_manager": <StateManager>,
    "full_refresh": False
}
```

The `.incremental_filter()` method reads this metadata to:
- Know which table is being built
- Access the state manager
- Get the last max value for filtering

## Running

```bash
# Install dependencies
uv sync

# Run example
uv run python example.py

# Test incremental processing
uv run python test_incremental.py
```

## Future Enhancements

- [ ] CLI interface with model selection (`pbt run +model+`)
- [ ] Full refresh flag (`pbt run --full-refresh`)
- [ ] Merge/upsert strategies (not just append)
- [ ] Partition-based incremental (delete+insert)
- [ ] Better date/datetime type handling in state
- [ ] Schema validation and testing hooks
- [ ] Parallel execution of independent models
- [ ] Support for multiple backends via Narwhals
