# Agent Instructions

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for all Python dependency management and virtual environment setup in this project.

### Why uv?

- **Fast**: Orders of magnitude faster than pip
- **Reliable**: Deterministic dependency resolution with lockfiles
- **Smart**: Automatically handles existing virtual environments
- **Simple**: Single tool for venv creation, package installation, and running scripts

### Common Commands

```bash
# Setup (works with new or existing venvs)
uv venv
uv sync

# Run scripts without activation
uv run python script.py

# Add dependencies
uv add package-name

# Update dependencies
uv sync
```

### For AI Agents

When working on this codebase:

1. **Always use `uv`** instead of `pip` or `python -m venv`
2. **Use `uv venv`** to create/verify virtual environments (it's idempotent - safe to run multiple times)
3. **Use `uv sync`** to install dependencies from `pyproject.toml`
4. **Use `uv run`** to execute Python scripts (no activation needed)
5. **Use the Makefile** targets (`make install`, `make run`, etc.) when available

### Examples

```bash
# Setup new environment
uv venv && uv sync

# Run tests
uv run pytest

# Run the example
uv run python example.py

# Or use make targets
make install
make run
make test
```

## Project Conventions

- Dependencies are managed in `pyproject.toml`
- Lock file is `uv.lock` (committed to repo)
- Virtual environment is `.venv/` (gitignored)
- Generated data files go in `data/` (gitignored)
