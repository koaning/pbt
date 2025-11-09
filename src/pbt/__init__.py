"""PBT - Polars-based transformation tool inspired by DBT"""

from pbt.core import conf
from pbt.extensions import setup_polars_extensions

# Setup Polars extensions on import
setup_polars_extensions()

__all__ = ["conf"]
