"""Core PBT functionality: conf, decorators, and execution"""

import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import polars as pl

from pbt.state import StateManager


class PBTApp:
    """Main PBT application that manages models and execution"""

    def __init__(self, root: str | Path = ".", env: str = "dev"):
        self.root = Path(root)
        self.env = env
        self.models: Dict[str, Dict[str, Any]] = {}
        self.state_manager = StateManager(self.root / ".pbt" / "state.json")

    def source(self, func: Callable) -> Callable:
        """Decorator for source tables - always recomputed"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Inject metadata
            if hasattr(result, '_pbt_metadata'):
                result._pbt_metadata = {"source": func.__name__, "type": "source"}
            return result

        self.models[func.__name__] = {
            "func": wrapper,
            "type": "source",
            "config": {}
        }
        return wrapper

    def model(self, func: Callable) -> Callable:
        """Decorator for view models - lazy evaluation, not materialized"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if hasattr(result, '_pbt_metadata'):
                result._pbt_metadata = {"source": func.__name__, "type": "model"}
            return result

        self.models[func.__name__] = {
            "func": wrapper,
            "type": "model",
            "config": {}
        }
        return wrapper

    def table(
        self,
        func: Optional[Callable] = None,
        *,
        incremental: bool = False,
        time_column: Optional[str] = None,
        unique_key: Optional[str | list[str]] = None,
        partition_by: Optional[str] = None
    ) -> Callable:
        """Decorator for materialized tables - writes to parquet"""
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def wrapper(*args, **kwargs):
                result = f(*args, **kwargs)
                if hasattr(result, '_pbt_metadata'):
                    result._pbt_metadata = {
                        "source": f.__name__,
                        "type": "table",
                        "target_table": f.__name__,
                        "state_manager": self.state_manager,
                        "full_refresh": False  # TODO: add CLI flag
                    }
                return result

            self.models[f.__name__] = {
                "func": wrapper,
                "type": "table",
                "config": {
                    "incremental": incremental,
                    "time_column": time_column,
                    "unique_key": unique_key,
                    "partition_by": partition_by
                }
            }
            return wrapper

        # Support both @app.table and @app.table(incremental=True)
        if func is None:
            return decorator
        else:
            return decorator(func)

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

    def run(self, target: Optional[str] = None, full_refresh: bool = False):
        """Execute models in dependency order"""
        execution_order = self.topological_sort()

        if target:
            # Only run target and its dependencies
            execution_order = [m for m in execution_order if m == target or target in self._get_downstream(m)]

        cache = {}

        for model_name in execution_order:
            model_info = self.models[model_name]
            func = model_info["func"]
            model_type = model_info["type"]
            config = model_info["config"]

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

            # Execute model
            print(f"Running {model_name} ({model_type})...")
            result = func(**kwargs)
            cache[model_name] = result

            # Materialize if it's a table
            if model_type == "table":
                output_path = self.root / "output" / f"{model_name}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                collected = result.collect()

                # Handle incremental append
                if config.get("incremental"):
                    if output_path.exists() and not full_refresh:
                        # Read existing data and append new records
                        existing = pl.read_parquet(output_path)
                        collected = pl.concat([existing, collected])
                        print(f"  -> Appended {len(result.collect())} new rows to {output_path}")
                    else:
                        print(f"  -> Wrote {output_path} (initial load)")
                else:
                    print(f"  -> Wrote {output_path}")

                # Write combined data
                collected.write_parquet(output_path)

                # Update state if incremental
                if config.get("incremental") and config.get("time_column"):
                    time_col = config["time_column"]
                    max_value = collected.select(time_col).max().item()
                    self.state_manager.update_state(model_name, {
                        "last_max_value": str(max_value),
                        "last_run": str(Path.cwd())  # TODO: proper timestamp
                    })

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


def conf(root: str | Path = ".", env: str = "dev") -> PBTApp:
    """Create a PBT application instance"""
    return PBTApp(root=root, env=env)
