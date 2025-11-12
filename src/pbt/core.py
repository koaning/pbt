"""Core PBT functionality: conf, decorators, and execution"""

import inspect
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import polars as pl

from pbt.state import StateManager


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
            if hasattr(result, 'collect'):
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
        output_path = self.app.root / "output" / f"{self.name}.parquet"
        if not output_path.exists():
            raise FileNotFoundError(
                f"Table '{self.name}' has not been materialized yet. Run app.run() first."
            )
        return pl.scan_parquet(output_path)


class PBTApp:
    """Main PBT application that manages models and execution"""

    def __init__(self, root: str | Path = ".", env: str = "dev"):
        self.root = Path(root)
        self.env = env
        self.models: Dict[str, Dict[str, Any]] = {}
        self.state_manager = StateManager(self.root / ".pbt" / "state.json")

    def source(self, func: Callable) -> Model:
        """Decorator for source tables - always recomputed"""
        model = Model(func, self, "source", {})
        self.models[func.__name__] = {
            "func": model,
            "type": "source",
            "config": {}
        }
        return model

    def model(self, func: Callable) -> Model:
        """Decorator for view models - lazy evaluation, not materialized"""
        model = Model(func, self, "model", {})
        self.models[func.__name__] = {
            "func": model,
            "type": "model",
            "config": {}
        }
        return model

    def table(
        self,
        func: Optional[Callable] = None,
        *,
        incremental: bool = False,
        time_column: Optional[str] = None,
        unique_key: Optional[str | list[str]] = None,
        partition_by: Optional[str] = None
    ) -> Model:
        """Decorator for materialized tables - writes to parquet"""
        def decorator(f: Callable) -> Model:
            config = {
                "incremental": incremental,
                "time_column": time_column,
                "unique_key": unique_key,
                "partition_by": partition_by
            }
            model = Model(f, self, "table", config)
            self.models[f.__name__] = {
                "func": model,
                "type": "table",
                "config": config
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
        partition_by: Optional[str] = None
    ) -> Model:
        """Sugar for incremental tables that auto-applies incremental_filter"""
        def decorator(func: Callable) -> Model:
            base_decorator = self.table(
                incremental=True,
                time_column=time_column,
                unique_key=unique_key,
                partition_by=partition_by
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

    def get_dependencies(self, func: Callable) -> list[str]:
        """Extract model dependencies from function parameters"""
        sig = inspect.signature(func)
        return [
            param.name for param in sig.parameters.values()
            if param.name in self.models
        ]

    def build_dag(self) -> Dict[str, list[str]]:
        """Build dependency graph from model definitions"""
        dag = {}
        for name, model_info in self.models.items():
            deps = self.get_dependencies(model_info["func"])
            dag[name] = deps
        return dag

    def topological_sort(self) -> list[str]:
        """Return models in execution order"""
        dag = self.build_dag()
        visited = set()
        result = []

        def visit(node: str):
            if node in visited:
                return
            visited.add(node)
            for dep in dag.get(node, []):
                visit(dep)
            result.append(node)

        for node in dag:
            visit(node)

        return result

    def run(self, target: Optional[str] = None, full_refresh: bool = False, debug: bool = False, silent: bool = False) -> Dict[str, Any]:
        """Execute models in dependency order and return execution cache"""
        execution_order = self.topological_sort()

        if debug and not silent:
            print(f"\n[DEBUG] Execution order: {' -> '.join(execution_order)}")
            print(f"[DEBUG] Full refresh: {full_refresh}")
            if target:
                print(f"[DEBUG] Target model: {target}")

        if target:
            # Only run target and its dependencies
            execution_order = [m for m in execution_order if m == target or target in self._get_downstream(m)]

        cache = {}

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
                    if hasattr(df, '_pbt_metadata'):
                        df._pbt_metadata = {
                            "target_table": model_name,
                            "state_manager": self.state_manager,
                            "full_refresh": full_refresh
                        }
                    kwargs[param_name] = df

            if debug and not silent:
                deps = list(kwargs.keys())
                print(f"\n[DEBUG] {model_name}:")
                print(f"  - Type: {model_type}")
                print(f"  - Dependencies: {deps if deps else 'none'}")
                if config.get("incremental"):
                    print(f"  - Incremental: True (time_column={config.get('time_column')})")
                    state = self.state_manager.get_state(model_name)
                    if state:
                        print(f"  - Last max value: {state.get('last_max_value', 'N/A')}")

            # Execute model
            if not silent:
                print(f"Running {model_name} ({model_type})...")
            result = func(**kwargs)
            cache[model_name] = result

            # Materialize if it's a table
            if model_type == "table":
                output_path = self.root / "output" / f"{model_name}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                collected = result.collect()

                if debug and not silent:
                    print(f"  - Collected {len(collected)} rows")

                # Handle incremental append
                if config.get("incremental"):
                    if output_path.exists() and not full_refresh:
                        # Read existing data and append new records
                        existing = pl.read_parquet(output_path)
                        new_rows = len(collected)
                        collected = pl.concat([existing, collected])
                        if debug and not silent:
                            print(f"  - Existing rows: {len(existing)}, New rows: {new_rows}, Total: {len(collected)}")
                        if not silent:
                            print(f"  -> Appended {new_rows} new rows to {output_path}")
                    else:
                        if not silent:
                            print(f"  -> Wrote {output_path} (initial load)")
                else:
                    if not silent:
                        print(f"  -> Wrote {output_path}")

                # Write combined data
                collected.write_parquet(output_path)

                # Update state metadata for observability
                state_payload = {"last_run": datetime.utcnow().isoformat()}
                if config.get("incremental") and config.get("time_column"):
                    time_col = config["time_column"]
                    max_value = collected.select(time_col).max().item()
                    state_payload["last_max_value"] = str(max_value)
                self.state_manager.update_state(model_name, state_payload)
                if debug and not silent:
                    meta_msg = ", ".join(f"{k}={v}" for k, v in state_payload.items())
                    print(f"  - Updated state: {meta_msg}")

        return cache

    def _get_downstream(self, model: str) -> set[str]:
        """Get all models that depend on this model"""
        dag = self.build_dag()
        downstream = set()

        def visit(node: str):
            for other, deps in dag.items():
                if node in deps and other not in downstream:
                    downstream.add(other)
                    visit(other)

        visit(model)
        return downstream

    def read_table(self, table_name: str) -> pl.DataFrame:
        """Read a materialized table from disk"""
        if table_name not in self.models:
            raise ValueError(f"Model '{table_name}' not found. Available: {list(self.models.keys())}")

        model_type = self.models[table_name]["type"]
        if model_type != "table":
            raise ValueError(f"'{table_name}' is a {model_type}, not a table. Use read_table() only for materialized tables.")

        output_path = self.root / "output" / f"{table_name}.parquet"
        if not output_path.exists():
            raise FileNotFoundError(f"Table '{table_name}' has not been materialized yet. Run app.run() first.")

        return pl.read_parquet(output_path)

    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata and function"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        return self.models[model_name]


def conf(root: str | Path = ".", env: str = "dev") -> PBTApp:
    """Create a PBT application instance"""
    return PBTApp(root=root, env=env)
