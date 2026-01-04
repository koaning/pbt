"""Metadata management for PBT models - column descriptions and schema handling"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import polars as pl


class SchemaChangeError(Exception):
    """Raised when schema changes are detected and safe=True"""

    pass


@dataclass
class ColumnMeta:
    """Metadata for a single column"""

    name: str
    dtype: str
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "description": self.description}


@dataclass
class TableMeta:
    """Metadata for a table including all columns"""

    name: str
    description: str = ""
    columns: list[ColumnMeta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "columns": [c.to_dict() for c in self.columns],
        }

    def __repr__(self) -> str:
        cols = ", ".join(f"{c.name}: {c.dtype}" for c in self.columns)
        return f"TableMeta({self.name}, columns=[{cols}])"


@dataclass
class SchemaChange:
    """Represents detected schema changes between stored and current schema"""

    added_columns: list[str] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)
    type_changes: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # col -> (old_type, new_type)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_columns or self.removed_columns or self.type_changes)

    def __str__(self) -> str:
        parts = []
        if self.added_columns:
            parts.append(f"added={self.added_columns}")
        if self.removed_columns:
            parts.append(f"removed={self.removed_columns}")
        if self.type_changes:
            type_strs = [
                f"{col}: {old}->{new}" for col, (old, new) in self.type_changes.items()
            ]
            parts.append(f"type_changes=[{', '.join(type_strs)}]")
        return ", ".join(parts) if parts else "no changes"


def dtype_to_string(dtype: pl.DataType) -> str:
    """Convert Polars dtype to human-readable string"""
    type_map = {
        pl.Utf8: "String",
        pl.String: "String",
        pl.Int8: "Int8",
        pl.Int16: "Int16",
        pl.Int32: "Int32",
        pl.Int64: "Int64",
        pl.UInt8: "UInt8",
        pl.UInt16: "UInt16",
        pl.UInt32: "UInt32",
        pl.UInt64: "UInt64",
        pl.Float32: "Float32",
        pl.Float64: "Float64",
        pl.Boolean: "Boolean",
        pl.Date: "Date",
        pl.Time: "Time",
        pl.Null: "Null",
    }

    # Check for exact type match
    for pl_type, name in type_map.items():
        if dtype == pl_type:
            return name

    # Handle parameterized types
    if isinstance(dtype, pl.Datetime):
        tz = dtype.time_zone
        return f"Datetime({tz})" if tz else "Datetime"
    if isinstance(dtype, pl.Duration):
        return "Duration"
    if isinstance(dtype, pl.List):
        inner = dtype_to_string(dtype.inner)
        return f"List[{inner}]"
    if isinstance(dtype, pl.Struct):
        return "Struct"

    # Fallback to string representation
    return str(dtype)


class SchemaManager:
    """Manages per-table YAML files containing schema, descriptions, and state.

    Each table gets its own file at .pbt/tables/{table_name}.yml containing:
    - name, description
    - columns with types and descriptions
    - state (last_run, last_max_value for incremental tables)
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.tables_dir = self.root / ".pbt" / "tables"
        self._cache: dict[str, dict] = {}

    def _table_path(self, table_name: str) -> Path:
        return self.tables_dir / f"{table_name}.yml"

    def has_schema(self, table_name: str) -> bool:
        """Check if a table file exists"""
        return self._table_path(table_name).exists()

    def _load_raw(self, table_name: str) -> dict:
        """Load raw YAML data for a table, using cache"""
        path = self._table_path(table_name)
        if not path.exists():
            return {}

        if table_name in self._cache:
            return self._cache[table_name]

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        self._cache[table_name] = data
        return data

    def _save_raw(self, table_name: str, data: dict) -> bool:
        """Save raw YAML data for a table. Returns True if new file."""
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self._table_path(table_name)
        is_new = not path.exists()

        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)

        # Update cache
        self._cache[table_name] = data
        return is_new

    def get_schema(self, table_name: str) -> Optional[TableMeta]:
        """Load schema from YAML file, returns None if not found"""
        data = self._load_raw(table_name)
        if not data:
            return None

        columns = [
            ColumnMeta(
                name=col["name"],
                dtype=col.get("dtype", ""),
                description=col.get("description", ""),
            )
            for col in data.get("columns", [])
        ]

        return TableMeta(
            name=data.get("name", table_name),
            description=data.get("description", ""),
            columns=columns,
        )

    def save_schema(
        self, table_name: str, polars_schema: dict[str, pl.DataType]
    ) -> bool:
        """Save/update schema, preserving existing descriptions and state.

        Returns True if this was a new file, False if updating existing.
        """
        path = self._table_path(table_name)
        is_new = not path.exists()

        # Load existing data to preserve descriptions and state
        existing = self._load_raw(table_name)
        existing_descriptions: dict[str, str] = {}
        for col in existing.get("columns", []):
            existing_descriptions[col["name"]] = col.get("description", "")

        # Build new schema with preserved descriptions
        columns = []
        for col_name, dtype in polars_schema.items():
            columns.append(
                {
                    "name": col_name,
                    "dtype": dtype_to_string(dtype),
                    "description": existing_descriptions.get(col_name, ""),
                }
            )

        # Preserve existing state and description
        data = {
            "name": table_name,
            "description": existing.get("description", ""),
            "columns": columns,
        }

        # Preserve state if it exists
        if "state" in existing:
            data["state"] = existing["state"]

        self._save_raw(table_name, data)
        return is_new

    def get_state(self, table_name: str) -> dict:
        """Get state for a table (last_run, last_max_value, etc.)"""
        data = self._load_raw(table_name)
        return data.get("state", {})

    def update_state(self, table_name: str, state: dict) -> None:
        """Update state for a table, preserving schema and descriptions"""
        data = self._load_raw(table_name)
        if not data:
            # Table file doesn't exist yet - this shouldn't happen normally
            # since save_schema is called before update_state
            data = {"name": table_name, "description": "", "columns": []}
        data["state"] = state
        self._save_raw(table_name, data)

    def clear_state(self, table_name: str) -> None:
        """Clear state for a table (force full refresh on next run)"""
        data = self._load_raw(table_name)
        if data and "state" in data:
            del data["state"]
            self._save_raw(table_name, data)

    def detect_changes(
        self, table_name: str, current_schema: dict[str, pl.DataType]
    ) -> SchemaChange:
        """Compare current schema against stored schema and return changes"""
        stored = self.get_schema(table_name)

        if stored is None:
            # No stored schema = no changes to detect (will be created)
            return SchemaChange()

        stored_cols = {col.name: col.dtype for col in stored.columns}
        current_cols = {
            name: dtype_to_string(dtype) for name, dtype in current_schema.items()
        }

        added = [name for name in current_cols if name not in stored_cols]
        removed = [name for name in stored_cols if name not in current_cols]
        type_changes = {
            name: (stored_cols[name], current_cols[name])
            for name in current_cols
            if name in stored_cols and stored_cols[name] != current_cols[name]
        }

        return SchemaChange(
            added_columns=added, removed_columns=removed, type_changes=type_changes
        )

    def build_table_meta(
        self, table_name: str, polars_schema: dict[str, pl.DataType]
    ) -> TableMeta:
        """Build TableMeta by combining Polars schema with stored descriptions"""
        stored = self.get_schema(table_name)

        # Build description lookup from stored schema
        descriptions: dict[str, str] = {}
        table_description = ""
        if stored:
            table_description = stored.description
            for col in stored.columns:
                descriptions[col.name] = col.description

        # Build columns from current schema with stored descriptions
        columns = [
            ColumnMeta(
                name=name,
                dtype=dtype_to_string(dtype),
                description=descriptions.get(name, ""),
            )
            for name, dtype in polars_schema.items()
        ]

        return TableMeta(
            name=table_name, description=table_description, columns=columns
        )
