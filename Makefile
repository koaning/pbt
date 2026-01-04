.PHONY: install run test clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies (creates or uses existing venv)"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove virtual environment and generated files"

install:
	@echo "Setting up virtual environment..."
	uv venv --allow-existing
	@echo "Installing dependencies..."
	uv sync --all-extras
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "Installation complete!"

test:
	uv run pytest tests/ -v

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf data/.pbt
	rm -rf data/output
	@echo "Clean complete!"
