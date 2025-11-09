.PHONY: install run test clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies (creates or uses existing venv)"
	@echo "  make clean      - Remove virtual environment and generated files"

install:
	@echo "Setting up virtual environment..."
	uv venv
	@echo "Installing dependencies..."
	uv sync
	@echo "Installation complete!"

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf data/.pbt
	rm -rf data/output
	@echo "Clean complete!"
