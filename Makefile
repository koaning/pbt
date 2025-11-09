.PHONY: install run test clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies (creates or uses existing venv)"
	@echo "  make run        - Run the example"
	@echo "  make test       - Run incremental processing test"
	@echo "  make clean      - Remove virtual environment and generated files"

install:
	@echo "Setting up virtual environment..."
	uv venv
	@echo "Installing dependencies..."
	uv sync
	@echo "Installation complete!"

run:
	uv run python example.py

test:
	uv run python test_incremental.py

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf data/.pbt
	rm -rf data/output
	@echo "Clean complete!"
