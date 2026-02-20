.PHONY: help install run-gui test-core test-vmd test-gui test-all lint format clean

# Default target shows available commands
help:
	@echo "TIFR-WORK Monorepo Commands"
	@echo "============================"
	@echo ""
	@echo "Development Commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make run-gui      - Launch the CustomTkinter GUI application"
	@echo "  make test-core    - Run pytest on autosim_core (ISOLATED - no GUI)"
	@echo "  make test-vmd     - Run pytest on vmd_plugins (ISOLATED - no GUI)"
	@echo "  make test-gui     - Run pytest on GUI frontend (explicit GUI tests)"
	@echo "  make test-all     - Run pytest on all packages (GUI tests skipped)"
	@echo "  make lint         - Run ruff linter across all directories"
	@echo "  make format       - Format code with ruff and black"
	@echo "  make clean        - Remove build artifacts and caches"
	@echo ""
	@echo "Note: On Windows, you may need to install make via chocolatey or use WSL"

# Install all dependencies including dev dependencies
install:
	pip install -e ".[all,dev]"

# Launch the CustomTkinter GUI application
run-gui:
	python -m gui_frontend.main

# Run pytest on the autosim_core compute engine (ISOLATED - no GUI)
test-core:
	pytest tests/autosim_core/ -v --cov=src/autosim_core --cov-report=term-missing

# Run pytest on vmd_plugins (ISOLATED - no GUI)
test-vmd:
	pytest tests/vmd_plugins/ -v --cov=src/vmd_plugins --cov-report=term-missing

# Run pytest on GUI (must be explicit to avoid accidental UI initialization)
test-gui:
	pytest tests/gui_frontend/ -v -m gui --cov=src/gui_frontend --cov-report=term-missing

# Run pytest on all packages (GUI tests skipped by default)
test-all:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run ruff linter across all directories
lint:
	ruff check src/ tests/
	@echo ""
	@echo "Linting complete! ✓"

# Format code with ruff and black
format:
	ruff check --fix src/ tests/
	black src/ tests/
	@echo ""
	@echo "Code formatting complete! ✓"

# Type checking with mypy
typecheck:
	mypy src/

# Run all quality checks (lint + typecheck + test)
check: lint typecheck test-all

# Clean build artifacts and caches
clean:
	@echo "Cleaning build artifacts and caches..."
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean complete! ✓"

# Initialize development environment
dev-setup:
	pip install --upgrade pip
	pip install -e ".[all,dev]"
	pre-commit install
	@echo ""
	@echo "Development environment setup complete! ✓"

# Run the autosim_core engine (standalone)
run-core:
	@echo "Running autosim_core simulation engine..."
	python -m autosim_core

# Build documentation (if using sphinx or similar)
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

# Watch and re-run tests on file changes (requires pytest-watch)
watch-test:
	pytest-watch tests/ -- -v

# Show project statistics
stats:
	@echo "Project Statistics:"
	@echo "==================="
	@echo ""
	@echo "Lines of code:"
	@find src -name "*.py" -type f -exec wc -l {} + | tail -1
	@echo ""
	@echo "Test files:"
	@find tests -name "*.py" -type f | wc -l
	@echo ""
	@echo "Packages:"
	@ls -d src/*/ | wc -l
