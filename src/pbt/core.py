"""Core PBT functionality: conf, decorators, and execution"""

import inspect
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import polars as pl

from pbt.exceptions import (
    PartitionSchemaError,
    CircularDependencyError,
    IncrementalConfigError,
)
from pbt.utils import coerce_datetime, parse_timestamp_value
from pbt.meta import SchemaManager, SchemaChangeError, TableMeta

if TYPE_CHECKING:
    from pbt.sinks.base import Sink


class Model:
    """Wrapper for PBT models that can be called as functions or accessed via .build()/.lazy()"""

    def __init__(self, func: Callable, app: "PBTApp", model_type: str, config: dict):
        self.func = func
        self.app = app
        self.name = func.__name__
        self.model_type = model_type
        self.config = config
        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        """Use as normal function"""
        return self.func(*args, **kwargs)

    def build(self) -> pl.DataFrame:
        """
        Execute and return result as eager DataFrame.
        For tables: Read materialized parquet
        For models/sources: Execute dependency chain and return result
        """
        if self.model_type == "table":
            return self.app.read_table(self.name)
        else:
            # Execute just this model and its dependencies
            cache = self.app.run(target=self.name, silent=True)
            result = cache.get(self.name)
            if result is None:
                raise RuntimeError(f"Model '{self.name}' was not executed successfully")
            # Collect if lazy
            if hasattr(result, "collect"):
                return result.collect()
            return result

    def build_lazy(self) -> pl.LazyFrame:
        """
        Lazy scan of materialized table - only works for tables.
        """
        if self.model_type != "table":
            raise ValueError(
                f"'{self.name}' is a {self.model_type}, not a table. "
                "Only materialized tables can be scanned lazily."
            )
        sink = self.config.get("sink") or self.app._default_sink
        if not sink.exists(self.name):
            raise FileNotFoundError(
                f"Table '{self.name}' has not been materialized yet. Run app.run() first."
            )
        return sink.read(self.name)

    def rerun(
        self,
        min_date: datetime | str,
        max_date: datetime | str,
        *,
        debug: bool = False,
        silent: bool = False,
    ) -> pl.DataFrame:
        """Reprocess a specific time window for incremental tables."""
        if self.model_type != "table":
            raise ValueError("Only materialized tables support rerun()")
        return self.app.rerun_table(
            self.name, min_date, max_date, debug=debug, silent=silent
        )

    def get_meta(self) -> TableMeta:
        """Get metadata for this model including column types and descriptions.

        Returns:
            TableMeta with column names, types (auto-detected from schema),
            and descriptions (from .pbt/tables/{name}.yml if available).

        For tables: Reads schema from materialized parquet file.
        For models/sources: Builds the model to get schema from result.
        """
        # Get Polars schema
        if self.model_type == "table":
            sink = self.config.get("sink") or self.app._default_sink
            if not sink.exists(self.name):
                raise FileNotFoundError(
                    f"Table '{self.name}' has not been materialized yet. "
                    "Run app.run() first to generate metadata."
                )
            # Read schema from the sink
            lf = sink.read(self.name)
            schema = lf.collect_schema()
        else:
            # For models/sources, we need to execute to get schema
            result = self.build()
            schema = result.schema

        return self.app.schema_manager.build_table_meta(self.name, schema)


class PBTApp:
    """Main PBT application that manages models and execution"""

    def __init__(self, root: str | Path = ".", env: str = "dev"):
        from pbt.sinks.local import LocalSink

        self.root = Path(root)
        self.env = env
        self.models: Dict[str, Dict[str, Any]] = {}
        self.schema_manager = SchemaManager(self.root)
        self._rerun_ranges: Dict[str, Tuple[str, str]] = {}
        # Default sink for backwards compatibility
        self._default_sink: "Sink" = LocalSink(self.root / "output")

    def source(self, func: Callable) -> Model:
        """Decorator for source tables - always recomputed"""
        model = Model(func, self, "source", {})
        self.models[func.__name__] = {"func": model, "type": "source", "config": {}}
        return model

    def model(self, func: Callable) -> Model:
        """Decorator for view models - lazy evaluation, not materialized"""
        model = Model(func, self, "model", {})
        self.models[func.__name__] = {"func": model, "type": "model", "config": {}}
        return model

    def table(
        self,
        func: Optional[Callable] = None,
        *,
        incremental: bool = False,
        time_column: Optional[str] = None,
        unique_key: Optional[str | list[str]] = None,
        partition_by: Optional[str | list[str]] = None,
        partition_mode: str = "append",
        lookback: Optional[timedelta] = None,
        sink: Optional["Sink"] = None,
    ) -> Model:
        """Decorator for materialized tables - writes to sink.

        Args:
            func: The function to decorate (for @app.table without parens)
            incremental: Whether this is an incremental table
            time_column: Column to use for incremental watermark
            unique_key: Column(s) for deduplication (future use)
            partition_by: Column(s) to partition by (Hive-style)
            partition_mode: "append" or "overwrite" for partition handling
            lookback: Lookback window for late-arriving data (incremental tables only)
            sink: Sink to write to (defaults to LocalSink at root/output)
        """

        def decorator(f: Callable) -> Model:
            # Normalize partition_by to list
            pb = None
            if partition_by is not None:
                pb = (
                    [partition_by]
                    if isinstance(partition_by, str)
                    else list(partition_by)
                )

            config = {
                "incremental": incremental,
                "time_column": time_column,
                "unique_key": unique_key,
                "partition_by": pb,
                "partition_mode": partition_mode,
                "lookback": lookback,
                "sink": sink,
            }
            model = Model(f, self, "table", config)
            self.models[f.__name__] = {
                "func": model,
                "type": "table",
                "config": config,
            }
            return model

        # Support both @app.table and @app.table(incremental=True)
        if func is None:
            return decorator
        else:
            return decorator(func)

    def incremental_table(
        self,
        *,
        time_column: str,
        unique_key: Optional[str | list[str]] = None,
        partition_by: str | list[str],
        partition_mode: str = "append",
        lookback: Optional[timedelta] = None,
        sink: Optional["Sink"] = None,
    ) -> Model:
        """Sugar for incremental tables that auto-applies incremental_filter.

        Incremental tables MUST be partitioned by date (derived from time_column).
        This ensures clean lookback semantics and prevents data duplication.

        Args:
            time_column: Column to use for incremental watermark (required)
            unique_key: Column(s) for deduplication (future use)
            partition_by: Column(s) to partition by (required, must include a Date column)
            partition_mode: "append" or "overwrite" for partition handling
            lookback: Optional lookback window to catch late-arriving data.
                      When set, reprocesses data from (last_max_value - lookback).
                      Partitions within the lookback window are overwritten.
            sink: Sink to write to (defaults to LocalSink at root/output)
        """

        def decorator(func: Callable) -> Model:
            base_decorator = self.table(
                incremental=True,
                time_column=time_column,
                unique_key=unique_key,
                partition_by=partition_by,
                partition_mode=partition_mode,
                lookback=lookback,
                sink=sink,
            )

            @wraps(func)
            def wrapped(*args, **kwargs):
                result = func(*args, **kwargs)
                # Automatically filter for new records if supported
                if hasattr(result, "incremental_filter"):
                    return result.incremental_filter(time_column)
                return result

            return base_decorator(wrapped)

        return decorator

    def rerun_table(
        self,
        table_name: str,
        min_date: datetime | str,
        max_date: datetime | str,
        *,
        debug: bool = False,
        silent: bool = False,
    ) -> pl.DataFrame:
        """Reprocess a specific window for an incremental table."""
        model_info = self.get_model(table_name)
        config = model_info["config"]

        if model_info["type"] != "table":
            raise ValueError(f"'{table_name}' is not a table")
        if not config.get("incremental") or not config.get("time_column"):
            raise ValueError(
                "rerun() is only available for incremental tables with a time_column"
            )

        start_dt = coerce_datetime(min_date, "min_date")
        end_dt = coerce_datetime(max_date, "max_date")
        if end_dt < start_dt:
            raise ValueError("max_date must be greater than or equal to min_date")

        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
        self._rerun_ranges[table_name] = (start_iso, end_iso)

        try:
            self.run(target=table_name, debug=debug, silent=silent)
        finally:
            self._rerun_ranges.pop(table_name, None)

        return self.read_table(table_name)

    def get_dependencies(self, func: Callable) -> list[str]:
        """Extract model dependencies from function parameters"""
        if isinstance(func, Model):
            target = func.func
        else:
            target = func
        sig = inspect.signature(target)
        return [
            param.name for param in sig.parameters.values() if param.name in self.models
        ]

    def build_dag(self) -> Dict[str, list[str]]:
        """Build dependency graph from model definitions"""
        dag = {}
        for name, model_info in self.models.items():
            deps = self.get_dependencies(model_info["func"])
            dag[name] = deps
        return dag

    def topological_sort(self) -> list[str]:
        """Return models in execution order.

        Raises:
            CircularDependencyError: If a cycle is detected in the dependency graph.
        """
        dag = self.build_dag()
        visited = set()
        in_progress = set()  # Track nodes currently being visited (on the stack)
        result = []

        def visit(node: str, path: list[str]):
            if node in visited:
                return
            if node in in_progress:
                # Found a cycle - build the cycle path for error message
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise CircularDependencyError(
                    f"Circular dependency detected: {' -> '.join(cycle)}"
                )

            in_progress.add(node)
            path.append(node)

            for dep in dag.get(node, []):
                visit(dep, path)

            path.pop()
            in_progress.remove(node)
            visited.add(node)
            result.append(node)

        for node in dag:
            visit(node, [])

        return result

    def run(
        self,
        target: Optional[str] = None,
        full_refresh: bool = False,
        safe: bool = True,
        debug: bool = False,
        silent: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute models in dependency order and return execution cache.

        Args:
            target: Only run this model and its dependencies
            full_refresh: Ignore existing data and rewrite from scratch
            safe: If True (default), raise SchemaChangeError when schema changes are detected
            debug: Print detailed debug information
            silent: Suppress all output
            dry_run: Show what would happen without actually writing

        Returns:
            Dict mapping model names to their results.
            If dry_run=True, also includes "dry_run_plans" key with write plans.
        """
        from pbt.sinks.base import DryRunPlan

        execution_order = self.topological_sort()

        if debug and not silent:
            print(f"\n[DEBUG] Execution order: {' -> '.join(execution_order)}")
            print(f"[DEBUG] Full refresh: {full_refresh}")
            print(f"[DEBUG] Dry run: {dry_run}")
            if target:
                print(f"[DEBUG] Target model: {target}")

        if target:
            needed = self._get_ancestors(target) | {target}
            execution_order = [m for m in execution_order if m in needed]

        cache = {}
        dry_run_plans: list[DryRunPlan] = []

        for model_name in execution_order:
            model_info = self.models[model_name]
            model_obj = model_info["func"]  # This is a Model instance
            model_type = model_info["type"]
            config = model_info["config"]

            # Get the actual function from the Model wrapper
            func = model_obj.func if isinstance(model_obj, Model) else model_obj

            # Prepare arguments by name matching
            sig = inspect.signature(func)
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in cache:
                    df = cache[param_name]
                    # Inject metadata for incremental operations
                    if hasattr(df, "_pbt_metadata"):
                        df._pbt_metadata = {
                            "target_table": model_name,
                            "schema_manager": self.schema_manager,
                            "full_refresh": full_refresh,
                            "reprocess_range": self._rerun_ranges.get(model_name),
                            "lookback": config.get("lookback"),
                        }
                    kwargs[param_name] = df

            if debug and not silent:
                deps = list(kwargs.keys())
                print(f"\n[DEBUG] {model_name}:")
                print(f"  - Type: {model_type}")
                print(f"  - Dependencies: {deps if deps else 'none'}")
                if config.get("incremental"):
                    print(
                        f"  - Incremental: True (time_column={config.get('time_column')})"
                    )
                    state = self.schema_manager.get_state(model_name)
                    if state:
                        print(
                            f"  - Last max value: {state.get('last_max_value', 'N/A')}"
                        )

            # Execute model
            if not silent:
                print(f"Running {model_name} ({model_type})...")
            result = func(**kwargs)
            cache[model_name] = result

            # Materialize if it's a table
            if model_type == "table":
                sink = config.get("sink") or self._default_sink
                partition_by = config.get("partition_by")
                partition_mode = config.get("partition_mode", "append")

                # Check for partition schema changes (unless full_refresh)
                if not full_refresh:
                    self._check_partition_schema(
                        model_name, partition_by, partition_mode
                    )

                collected = result.collect()

                if debug and not silent:
                    print(f"  - Collected {len(collected)} rows")

                # Validate time_column for incremental tables
                if config.get("incremental") and config.get("time_column"):
                    self._validate_time_column(
                        model_name, collected.schema, config["time_column"]
                    )
                    # Also validate date partition requirement for incremental tables
                    if partition_by:
                        self._validate_date_partition(
                            model_name, collected.schema, partition_by
                        )

                # Check for schema changes before writing
                current_schema = collected.schema
                changes = self.schema_manager.detect_changes(model_name, current_schema)

                if changes.has_changes:
                    if safe:
                        raise SchemaChangeError(
                            f"Schema change detected for '{model_name}': {changes}\n"
                            f"Run with safe=False to proceed, or update .pbt/tables/{model_name}.yml"
                        )
                    else:
                        if not silent:
                            print(f"  Warning: Schema change: {changes}")

                reprocess_range = self._rerun_ranges.get(model_name)

                # Handle incremental merge with existing data
                if config.get("incremental") and not full_refresh:
                    collected = self._apply_incremental_merge(
                        model_name,
                        collected,
                        config,
                        sink,
                        reprocess_range,
                        debug,
                        silent,
                    )

                # Determine effective partition mode for write
                # When lookback is set, use overwrite mode for affected partitions
                # to avoid duplicating data in the lookback window
                effective_partition_mode = partition_mode
                if config.get("lookback") and partition_by:
                    effective_partition_mode = "overwrite"

                # Write via sink
                write_result = sink.write(
                    df=collected,
                    name=model_name,
                    partition_by=partition_by,
                    partition_mode=effective_partition_mode,
                    dry_run=dry_run,
                )

                if dry_run:
                    dry_run_plans.append(write_result)
                    if not silent:
                        self._print_dry_run_plan(write_result)
                else:
                    if not silent:
                        self._print_write_result(model_name, write_result, config)

                    # Update/create schema file (preserves existing descriptions)
                    is_new_schema = self.schema_manager.save_schema(
                        model_name, current_schema
                    )
                    if is_new_schema and not silent:
                        print(f"  -> Generated schema: .pbt/tables/{model_name}.yml")

                    # Update state metadata
                    self._update_table_state(
                        model_name, collected, config, partition_by, partition_mode
                    )
                    if debug and not silent:
                        state = self.schema_manager.get_state(model_name)
                        meta_msg = ", ".join(f"{k}={v}" for k, v in state.items())
                        print(f"  - Updated state: {meta_msg}")

        if dry_run:
            cache["dry_run_plans"] = dry_run_plans

        return cache

    def _check_partition_schema(
        self,
        model_name: str,
        partition_by: Optional[list[str]],
        partition_mode: str,
    ) -> None:
        """Check for partition schema changes and error if mismatch."""
        state = self.schema_manager.get_state(model_name)
        if not state:
            return  # No existing state, nothing to check

        stored_partition_by = state.get("partition_by")
        stored_partition_mode = state.get("partition_mode")

        # Only check if we have stored partition config
        if stored_partition_by is not None or stored_partition_mode is not None:
            current_pb = partition_by or []
            stored_pb = stored_partition_by or []

            if current_pb != stored_pb:
                raise PartitionSchemaError(
                    f"Partition schema changed for '{model_name}'.\n"
                    f"  Stored: partition_by={stored_pb}\n"
                    f"  Current: partition_by={current_pb}\n"
                    f"Run with full_refresh=True to rewrite the table."
                )

            if stored_partition_mode and partition_mode != stored_partition_mode:
                raise PartitionSchemaError(
                    f"Partition mode changed for '{model_name}'.\n"
                    f"  Stored: partition_mode={stored_partition_mode}\n"
                    f"  Current: partition_mode={partition_mode}\n"
                    f"Run with full_refresh=True to confirm this change."
                )

    def _validate_time_column(
        self,
        model_name: str,
        schema: dict[str, pl.DataType],
        time_column: str,
    ) -> None:
        """Validate that time_column exists and is a datetime-like type."""
        if time_column not in schema:
            available = list(schema.keys())
            raise IncrementalConfigError(
                f"time_column '{time_column}' not found in '{model_name}'.\n"
                f"  Available columns: {available}"
            )

        col_type = schema[time_column]
        # Check for datetime-like types
        valid_types = (pl.Datetime, pl.Date)
        if not isinstance(col_type, valid_types):
            raise IncrementalConfigError(
                f"time_column '{time_column}' in '{model_name}' must be Datetime or Date type.\n"
                f"  Got: {col_type}\n"
                f"  Hint: Use .cast(pl.Datetime) or .str.to_datetime() to convert."
            )

    def _validate_date_partition(
        self,
        model_name: str,
        schema: dict[str, pl.DataType],
        partition_by: list[str],
    ) -> None:
        """Validate that at least one partition column is a Date type."""
        date_partitions = [
            col
            for col in partition_by
            if col in schema and isinstance(schema[col], pl.Date)
        ]

        if not date_partitions:
            partition_types = {
                col: str(schema.get(col, "missing")) for col in partition_by
            }
            raise IncrementalConfigError(
                f"Incremental table '{model_name}' requires at least one Date partition.\n"
                f"  partition_by columns: {partition_types}\n"
                f"  Hint: Add a date column derived from time_column:\n"
                f"    .with_columns(pl.col('timestamp').dt.date().alias('date'))"
            )

    def _apply_incremental_merge(
        self,
        model_name: str,
        new_data: pl.DataFrame,
        config: dict,
        sink,
        reprocess_range: Optional[Tuple[str, str]],
        debug: bool,
        silent: bool,
    ) -> pl.DataFrame:
        """Merge new data with existing for incremental tables."""
        if not sink.exists(model_name):
            return new_data

        partition_by = config.get("partition_by")
        time_column = config.get("time_column")

        # For partitioned tables, let the sink handle the merge logic
        # (both append and overwrite modes - sink knows how to handle each)
        if partition_by:
            return new_data

        # For non-partitioned or append mode: merge with existing data
        new_rows = len(new_data)
        existing = sink.read(model_name).collect()

        # Remove reprocessed rows if applicable
        if reprocess_range and time_column:
            start_value, end_value = reprocess_range
            start_literal = pl.lit(parse_timestamp_value(start_value))
            end_literal = pl.lit(parse_timestamp_value(end_value))
            existing = existing.filter(
                (pl.col(time_column) < start_literal)
                | (pl.col(time_column) > end_literal)
            )

        collected = pl.concat([existing, new_data])

        # Sort by time column if reprocessing
        if reprocess_range and time_column:
            collected = collected.sort(time_column)

        if debug and not silent:
            print(
                f"  - Existing rows: {len(existing)}, New rows: {new_rows}, Total: {len(collected)}"
            )

        return collected

    def _update_table_state(
        self,
        model_name: str,
        data: pl.DataFrame,
        config: dict,
        partition_by: Optional[list[str]],
        partition_mode: str,
    ) -> None:
        """Update state with partition metadata."""
        state_payload = {
            "last_run": datetime.now(UTC).isoformat(),
            "partition_by": partition_by,
            "partition_mode": partition_mode,
        }

        if config.get("incremental") and config.get("time_column"):
            time_col = config["time_column"]
            max_value = data.select(time_col).max().item()
            state_payload["last_max_value"] = str(max_value)

        self.schema_manager.update_state(model_name, state_payload)

    def _print_dry_run_plan(self, plan) -> None:
        """Print dry-run plan details."""
        print(f"\n[DRY RUN] {plan.table_name}")
        print(f"  Destination: {plan.destination}")
        print(f"  Rows to write: {plan.rows_to_write}")
        if plan.partitions_affected:
            print(f"  Partitions affected ({len(plan.partitions_affected)}):")
            for p in plan.partitions_affected[:10]:  # Limit output
                op = plan.partition_operations.get(p, "unknown")
                print(f"    - {p} [{op}]")
            if len(plan.partitions_affected) > 10:
                print(f"    ... and {len(plan.partitions_affected) - 10} more")

    def _print_write_result(self, model_name: str, result, config: dict) -> None:
        """Print write result summary."""
        if result.partitions_affected:
            print(
                f"  -> Wrote {result.rows_written} rows to {len(result.partitions_affected)} partitions"
            )
        else:
            op = "initial load" if result.operation == "create" else result.operation
            print(f"  -> Wrote {result.destination} ({op})")

    def _get_ancestors(self, model: str) -> set[str]:
        """Get all upstream dependencies for a model"""
        dag = self.build_dag()
        ancestors = set()

        def visit(node: str):
            for dep in dag.get(node, []):
                if dep not in ancestors:
                    ancestors.add(dep)
                    visit(dep)

        visit(model)
        return ancestors

    def read_table(self, table_name: str) -> pl.DataFrame:
        """Read a materialized table from sink."""
        if table_name not in self.models:
            raise ValueError(
                f"Model '{table_name}' not found. Available: {list(self.models.keys())}"
            )

        model_info = self.models[table_name]
        if model_info["type"] != "table":
            raise ValueError(
                f"'{table_name}' is a {model_info['type']}, not a table. "
                "Use read_table() only for materialized tables."
            )

        sink = model_info["config"].get("sink") or self._default_sink
        if not sink.exists(table_name):
            raise FileNotFoundError(
                f"Table '{table_name}' has not been materialized yet. Run app.run() first."
            )

        return sink.read(table_name).collect()

    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata and function"""
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.models.keys())}"
            )
        return self.models[model_name]


def conf(root: str | Path = ".", env: str = "dev") -> PBTApp:
    """Create a PBT application instance"""
    return PBTApp(root=root, env=env)
