"""Tests for schema change detection."""

import polars as pl
import pytest

from pbt import conf, SchemaChangeError


def test_detect_schema_change_added_column(tmp_path):
    """Schema change detection should catch added columns."""
    app = conf(root=tmp_path)

    @app.source
    def data():
        return pl.LazyFrame({"id": [1, 2], "value": ["a", "b"]})

    @app.table
    def output(data):
        return data

    # First run - establishes schema
    app.run(silent=True)

    # Redefine with extra column
    app.models.clear()

    @app.source
    def data():
        return pl.LazyFrame({"id": [1, 2], "value": ["a", "b"], "new_col": [1.0, 2.0]})

    @app.table
    def output(data):
        return data

    # Second run should detect added column and raise
    with pytest.raises(SchemaChangeError, match="added="):
        app.run(silent=True)


def test_safe_false_allows_schema_change(tmp_path):
    """safe=False should allow schema changes to proceed."""
    app = conf(root=tmp_path)

    @app.source
    def data():
        return pl.LazyFrame({"id": [1, 2], "value": ["a", "b"]})

    @app.table
    def output(data):
        return data

    # First run
    app.run(silent=True)

    # Redefine with extra column
    app.models.clear()

    @app.source
    def data():
        return pl.LazyFrame({"id": [1, 2], "value": ["a", "b"], "extra": [True, False]})

    @app.table
    def output(data):
        return data

    # With safe=False, should succeed
    app.run(silent=True, safe=False)

    # Verify data was written with new column
    result = app.read_table("output")
    assert "extra" in result.columns
