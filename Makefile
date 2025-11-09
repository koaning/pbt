.PHONY: install run test clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies (creates or uses existing venv)"
	@echo "  make clean      - Remove virtual environment and generated files"

install:
	@echo "Setting up virtual environment..."
	uv venv --allow-existing
	@echo "Installing dependencies..."
	uv sync --extra dev
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "Installation complete!"

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf data/.pbt
	rm -rf data/output
	@echo "Clean complete!"
