# TIFR-WORK

Monte Carlo Simulation Suite for Protein Folding Dynamics with VMD Integration

## Overview

TIFR-WORK is a monorepo containing three integrated packages for computational molecular dynamics:

- **autosim_core**: Core Monte Carlo simulation engine for protein folding dynamics
- **vmd_plugins**: VMD (Visual Molecular Dynamics) integration and visualization tools
- **gui_frontend**: CustomTkinter-based GUI for visualization and control

## Project Structure

```
TIFR-WORK/
├── src/
│   ├── autosim_core/         # Core simulation engine (Level 0 - no dependencies)
│   │   └── __init__.py
│   ├── vmd_plugins/          # VMD integration (Level 1 - depends on autosim_core)
│   │   └── __init__.py
│   └── gui_frontend/         # GUI application (Level 2 - depends on both)
│       ├── __init__.py
│       └── main.py
├── tests/                    # Test suite
│   └── __init__.py
├── docs/                     # Documentation
│   └── README.md
├── pyproject.toml           # Project configuration and dependencies
├── Makefile                 # Build and development commands
└── README.md               # This file
```

## Dependency Architecture

The project follows a strict hierarchical dependency structure to prevent circular dependencies:

```
autosim_core (Core computation engine)
    ↑
    │
vmd_plugins (depends on autosim_core)
    ↑
    │
gui_frontend (depends on autosim_core and vmd_plugins)
```

**Zero circular dependencies by design.**

📖 **See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architectural guidelines and common pitfalls to avoid.**

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) make utility for using Makefile commands

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vihaankulkarni29/TIFR-Work.git
cd TIFR-WORK
```

2. Install the package with all dependencies:
```bash
pip install -e ".[all,dev]"
```

Or use the Makefile:
```bash
make install
```

### Development Setup

For a complete development environment with pre-commit hooks:
```bash
make dev-setup
```

## Usage

### Running the GUI Application

```bash
make run-gui
```

Or directly:
```bash
python -m gui_frontend.main
```

### Running Tests

Test the core simulation engine (isolated, no GUI):
```bash
make test-core
```

Test VMD plugins (isolated, no GUI):
```bash
make test-vmd
```

Test all packages (GUI tests skipped by default to prevent accidental UI initialization):
```bash
make test-all
```

Test GUI explicitly (when needed):
```bash
make test-gui
```

**Note:** Tests are organized in isolated directories to prevent accidental monolith execution. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

### Code Quality

Run linter (ruff):
```bash
make lint
```

Format code:
```bash
make format
```

Type checking:
```bash
make typecheck
```

Run all quality checks:
```bash
make check
```

## Makefile Commands

Available commands in the Makefile:

- `make help` - Show all available commands
- `make install` - Install all dependencies
- `make run-gui` - Launch the CustomTkinter GUI application
- `make test-core` - Run pytest on autosim_core (isolated, no GUI)
- `make test-vmd` - Run pytest on vmd_plugins (isolated, no GUI)
- `make test-gui` - Run pytest on GUI frontend (explicit GUI tests)
- `make test-all` - Run pytest on all packages (GUI tests skipped by default)
- `make lint` - Run ruff linter across all directories
- `make format` - Format code with ruff and black
- `make clean` - Remove build artifacts and caches
- `make dev-setup` - Initialize development environment

## Package Details

### autosim_core

Core computational engine for Monte Carlo simulations. This package is completely independent and has no dependencies on other project packages.

**Dependencies:** numpy, scipy, numba, pandas, h5py

### vmd_plugins

VMD integration tools and molecular visualization utilities. Depends on autosim_core for simulation data access.

**Dependencies:** autosim_core, matplotlib, mdanalysis

### gui_frontend

CustomTkinter-based graphical user interface for controlling simulations and visualizing results.

**Dependencies:** autosim_core, vmd_plugins, customtkinter, pillow, matplotlib

## Development Guidelines

### Adding New Features

1. Identify the appropriate package (autosim_core, vmd_plugins, or gui_frontend)
2. Ensure new code respects the dependency hierarchy
3. Add corresponding tests in the `tests/` directory
4. Update documentation as needed

### Code Style

This project uses:
- **ruff** for linting and import sorting
- **black** for code formatting
- **mypy** for type checking
- **pytest** for testing

Run `make check` before committing to ensure code quality.

### Testing

Write tests for all new functionality. Mark tests appropriately:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Contact

Vihaan Kulkarni - [GitHub](https://github.com/vihaankulkarni29)

Project Link: [https://github.com/vihaankulkarni29/TIFR-Work](https://github.com/vihaankulkarni29/TIFR-Work)
