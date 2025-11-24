# PBT Design Document

## Overview

PBT (Polars-Based Transformations) is a lightweight alternative to DBT that works with Polars/Narwhals LazyFrames instead of SQL.

## Core Design Decisions

### 1. Metadata via Monkeypatch

**Problem**: How does `.incremental_filter()` know which table it's building and what the last max value was?

**Solution**: Monkeypatch `pl.LazyFrame._pbt_metadata` to carry context through the pipeline.

```python
# During execution, PBT injects:
dataframe._pbt_metadata = {
    "target_table": "user_events",
    "state_manager": <StateManager instance>,
    "full_refresh": False,
    "reprocess_range": ("2025-01-15T12:00:00", "2025-01-15T18:00:00")  # optional override
}

# Then .incremental_filter() reads this
def incremental_filter(self, column):
    meta = self._pbt_metadata
    rerun_range = meta.get("reprocess_range")
    if rerun_range:
        start, end = rerun_range
        return self.filter(pl.col(column).is_between(start, end, closed="both"))
    last_val = meta["state_manager"].get_state(meta["target_table"])["last_max_value"]
    return self.filter(pl.col(column) > last_val)
```

**Why this approach?**
- ✅ No magic global state
- ✅ No extra function parameters
- ✅ Clean user API
- ✅ Metadata travels with the dataframe
- ⚠️ Relies on monkeypatching (but acceptable for this use case)

### 2. Dependency Resolution via Inspection

**Problem**: How to build a DAG without explicit `ref()` calls?

**Solution**: Inspect function signatures and match parameter names to model names.

```python
@app.model
def model_b(source_a):  # "source_a" matches the function name
    return source_a.filter(...)

# PBT does:
sig = inspect.signature(model_b)
dependencies = [p for p in sig.parameters if p in app.models]
# dependencies = ["source_a"]
```

**Why this approach?**
- ✅ Clean, Pythonic API
- ✅ Type hints could enable better validation
- ✅ No string-based references
- ⚠️ Parameter names must match model names exactly

### 3. Incremental Strategy: Append-Only

**Current implementation**: Simple append based on time column

```python
@app.incremental_table(time_column="timestamp")
def events(raw_logs):
    return raw_logs.select(...)
```

**Flow**:
1. Read state: `last_max_value = "2025-11-05 23:00:00"`
2. Automatically filter: `WHERE timestamp > '2025-11-05 23:00:00'`
3. Collect new records
4. Append to existing parquet
5. Update state with new max value

**Why start with append-only?**
- ✅ Simplest incremental strategy
- ✅ Covers most use cases (event logs, timeseries)
- ✅ Foundation for more complex strategies

**Future strategies**:
- **Merge/Upsert**: Use unique_key to replace existing rows
- **Partition**: Delete and rewrite specific partitions
- **Snapshot**: Full refresh with versioning

### 4. State Management

**Format**: Simple JSON file at `{root}/.pbt/state.json`

```json
{
  "model_name": {
    "last_run": "2025-11-05T23:15:42.123456"
  },
  "incremental_model": {
    "last_run": "2025-11-05T23:15:42.123456",
    "last_max_value": "2025-11-05 23:00:00"
  }
}
```

Each materialized table records at least a `last_run` UTC timestamp. Incremental tables store an additional `last_max_value` watermark for their configured time column.

**Why JSON?**
- ✅ Human-readable
- ✅ Easy to debug/edit
- ✅ No external dependencies
- ⚠️ Not suitable for concurrent writes (future: use SQLite or locking)

### 5. Materialization

**Tables**: Write to `{root}/output/{model_name}.parquet`

**Incremental append logic**:
```python
collected = result.collect()

if incremental and file_exists and not full_refresh:
    existing = pl.read_parquet(file)
    collected = pl.concat([existing, collected])

collected.write_parquet(file)
```

When `reprocess_range` is set (via `Model.rerun()`), existing rows within that window are removed before concatenation and the combined result is resorted on the incremental column.

**Why parquet?**
- ✅ Columnar format, efficient for analytics
- ✅ Native Polars support
- ✅ Portable across tools
- ⚠️ Single file per table (future: partitioned parquet)

## Key Differences from DBT

| Feature | DBT | PBT |
|---------|-----|-----|
| Query language | SQL | Python + Polars |
| Model definition | `.sql` files | Python decorators |
| Dependency refs | `{{ ref('model') }}` | Function parameters |
| Execution | Eager (immediate SQL) | Lazy (LazyFrame) |
| State | Manifest files | Simple JSON |
| Incremental | SQL-based strategies | DataFrame filters |
| Testing | Built-in tests | User-defined functions |

## Trade-offs

### Advantages
- **Lazy evaluation**: Entire pipeline is lazy until materialization
- **Type safety**: Python types + potential for static analysis
- **Flexibility**: Full power of Polars/Python
- **Simplicity**: ~300 lines of core code vs DBT's complexity

### Limitations
- **No SQL**: Can't leverage existing SQL skills/code
- **No built-in testing**: Users must write their own checks
- **No packages/hub**: DBT has extensive ecosystem
- **Early stage**: Missing many DBT features (snapshots, seeds, docs)

### Rerun / Backfill Windows

- `Model.rerun(min_date, max_date)` registers a temporary `reprocess_range` on the target table.
- Bounds must be ISO-8601 strings or `datetime` objects so we can compare and remove the matching window from existing parquet files.
- Dependencies receive this metadata so `.incremental_filter()` narrows to the explicit bounds instead of `last_max_value`.
- During materialization we drop rows from the historical window, insert the recomputed slice, and resort on the time column to keep chronological order.
- State tracking continues to store the overall `last_max_value` along with the latest `last_run` timestamp.

## Future Enhancements

### Near-term
1. **CLI**: `pbt run --select +model+`
2. **Full refresh**: `pbt run --full-refresh model`
3. **Better type handling**: Parse dates/ints correctly in state
4. **Error handling**: Better messages, rollback on failure

### Medium-term
1. **Merge strategy**: Upsert based on unique keys
2. **Partition strategy**: Delete+insert by partition
3. **Parallel execution**: Independent models run concurrently
4. **Schema validation**: Check output schemas match expectations
5. **Testing hooks**: Built-in assertions framework

### Long-term
1. **Narwhals integration**: Support pandas, DuckDB, etc.
2. **Streaming**: Incremental processing of streaming data
3. **Observability**: Lineage tracking, metrics, monitoring
4. **Collaboration**: Share models, documentation generation

## Implementation Notes

### Metadata Propagation
Currently metadata must be re-injected for each model execution. This is because Polars operations don't preserve custom attributes:

```python
df = df.filter(...)  # ._pbt_metadata is lost
```

We re-inject before calling each model function.

### State Serialization
Datetime values are serialized as strings:
```python
str(datetime(2025, 11, 5, 23, 0))  # "2025-11-05 23:00:00"
```

Then parsed back when filtering:
```python
datetime.fromisoformat(value.replace(" ", "T"))
```

This is fragile and should be improved to handle:
- Dates vs datetimes
- Timezones
- Numeric watermarks (IDs, version numbers)

### DAG Execution
Simple topological sort. No parallelization yet, but the architecture supports it:

```python
# Future: identify independent models
independent = [m for m in dag if not dag[m]]
asyncio.gather(*[run_model(m) for m in independent])
```

## Open Questions

1. **Narwhals priority?** Should we use Narwhals from the start or optimize for Polars first?
2. **Testing framework?** Provide built-in assertions or leave to users?
3. **Snapshot strategy?** How to handle slowly changing dimensions?
4. **Multi-backend state?** Different state storage for different environments?
5. **Backwards compatibility?** How to handle breaking changes to state format?
