# TIFR-WORK Architecture Documentation

## Core Principle: Zero Circular Dependencies

This monorepo is designed with a **strict hierarchical dependency structure** to ensure maintainability, testability, and architectural clarity.

## Dependency Hierarchy

```
┌─────────────────────────────────────────┐
│  Level 0: autosim_core                  │
│  (Core computation engine)              │
│  • Monte Carlo simulations              │
│  • Physics calculations                 │
│  • Statistical analysis                 │
│  • NO external package dependencies     │
└─────────────────────────────────────────┘
                  ↑
                  │ (can import)
                  │
┌─────────────────────────────────────────┐
│  Level 1: vmd_plugins                   │
│  (VMD integration & visualization)      │
│  • CAN import from: autosim_core        │
│  • CANNOT import from: gui_frontend     │
└─────────────────────────────────────────┘
                  ↑
                  │ (can import)
                  │
┌─────────────────────────────────────────┐
│  Level 2: gui_frontend                  │
│  (User interface)                       │
│  • CAN import from: autosim_core        │
│  • CAN import from: vmd_plugins         │
└─────────────────────────────────────────┘
```

**This design makes circular dependencies structurally impossible.**

---

## Common Pitfalls & How We Avoid Them

### ❌ PITFALL #1: The "Utils" Dumpster

**BAD PRACTICE:**
```python
# ❌ NEVER DO THIS
# utils.py (at root)
def shared_function():
    pass

# autosim_core/engine.py
from utils import shared_function  # ❌ Bad!

# gui_frontend/app.py
from utils import shared_function  # ❌ Creates shared dependency!
```

**CORRECT APPROACH:**
```python
# ✅ DO THIS INSTEAD
# src/autosim_core/helpers.py
def shared_function():
    """Shared logic belongs in the core engine."""
    pass

# src/autosim_core/__init__.py
from .helpers import shared_function
__all__ = ['shared_function']

# src/gui_frontend/app.py
from autosim_core import shared_function  # ✅ GUI imports from core!
```

**RULE:** If a function is needed by both the GUI and the compute engine, it belongs in `autosim_core`. The GUI can import it, but the engine must **never** import from the GUI.

### ❌ PITFALL #2: Accidental Monolith Execution

**PROBLEM:**
Running `pytest` at the root without proper configuration could attempt to spin up the CustomTkinter UI while testing the AutoSIM physics math.

**SOLUTION:**
We've implemented **strictly partitioned tests**:

```
tests/
├── autosim_core/        # Physics tests - NO GUI imports
│   ├── __init__.py
│   └── test_core.py
├── vmd_plugins/         # VMD tests - NO GUI imports
│   ├── __init__.py
│   └── test_vmd.py
└── gui_frontend/        # GUI tests - Isolated with @pytest.mark.gui
    ├── __init__.py
    └── test_gui.py
```

**Test Isolation Commands:**

```bash
# Test ONLY the core engine (no GUI)
make test-core
# OR
pytest tests/autosim_core/ -v

# Test ONLY vmd plugins (no GUI)
pytest tests/vmd_plugins/ -v

# Test everything (GUI tests are skipped by default)
make test-all
# OR
pytest tests/ -v

# Test ONLY GUI (when you explicitly want to)
pytest tests/gui_frontend/ -v -m gui
```

**Pytest Configuration:**
- GUI tests are marked with `@pytest.mark.gui` and automatically skipped
- Tests are organized by module in separate directories
- Each test module verifies it doesn't accidentally import GUI code

---

## Package Design Rules

### autosim_core (Level 0)

**Purpose:** Pure computational engine for Monte Carlo simulations

**Rules:**
- ✅ Can import: `numpy`, `scipy`, `numba`, `pandas`, `h5py`
- ❌ CANNOT import: `gui_frontend`, `vmd_plugins`, `customtkinter`
- ✅ Should contain: All shared business logic and utilities
- ✅ Should be: Completely testable without a display/GUI

**Design Pattern:**
```python
# autosim_core should be pure Python + scientific libraries
# No GUI dependencies whatsoever
```

### vmd_plugins (Level 1)

**Purpose:** VMD integration and molecular visualization

**Rules:**
- ✅ Can import: `autosim_core`, `matplotlib`, `mdanalysis`
- ❌ CANNOT import: `gui_frontend`, `customtkinter`
- ✅ Should contain: VMD interfaces, trajectory processors
- ✅ Should be: Testable without GUI

**Design Pattern:**
```python
from autosim_core import SimulationEngine  # ✅ Allowed

# Use core functionality and add visualization
class TrajectoryAnalyzer:
    def __init__(self, simulation_data):
        self.data = simulation_data
```

### gui_frontend (Level 2)

**Purpose:** User interface and visualization

**Rules:**
- ✅ Can import: `autosim_core`, `vmd_plugins`, `customtkinter`
- ✅ Should contain: UI logic, event handlers, displays
- ✅ Should be: The only package with GUI dependencies

**Design Pattern:**
```python
from autosim_core import SimulationEngine  # ✅ Import core
from vmd_plugins import TrajectoryAnalyzer  # ✅ Import plugins

class TIFRWorkApp:
    def __init__(self):
        self.engine = SimulationEngine()  # Use core
        self.analyzer = TrajectoryAnalyzer()  # Use plugins
```

---

## Shared Code Strategy

### When Code Needs to Be Shared

**Decision Tree:**

1. **Is it computational/business logic?**
   - **YES** → Put it in `autosim_core`
   - **NO** → Continue to #2

2. **Is it VMD-specific visualization/analysis?**
   - **YES** → Put it in `vmd_plugins`
   - **NO** → Continue to #3

3. **Is it UI-specific?**
   - **YES** → Put it in `gui_frontend`
   - **NO** → Probably belongs in `autosim_core`

**Examples:**

| Function | Belongs In | Reasoning |
|----------|------------|-----------|
| `calculate_energy_distribution()` | `autosim_core` | Pure physics calculation |
| `monte_carlo_step()` | `autosim_core` | Core algorithm |
| `load_pdb_file()` | `vmd_plugins` | Molecular file handling |
| `render_trajectory()` | `vmd_plugins` | Visualization |
| `create_button()` | `gui_frontend` | UI component |
| `validate_input_range()` | `autosim_core` | Business logic (even if used by GUI) |

---

## Testing Strategy

### Test Isolation Guarantees

Each test suite includes **import guards** to prevent accidental GUI loading:

```python
# tests/autosim_core/test_core.py
def test_no_gui_imports():
    """Verify that autosim_core does not import GUI packages."""
    import autosim_core
    import sys
    
    assert 'gui_frontend' not in sys.modules
    assert 'customtkinter' not in sys.modules
```

### Test Organization

```
tests/
├── autosim_core/
│   └── test_core.py         # Tests core physics (unit tests)
├── vmd_plugins/
│   └── test_vmd.py          # Tests VMD integration (integration tests)
└── gui_frontend/
    └── test_gui.py          # Tests UI (marked with @pytest.mark.gui, skipped by default)
```

### Running Tests Safely

```bash
# Safe: Tests only physics
pytest tests/autosim_core/

# Safe: Tests only VMD plugins
pytest tests/vmd_plugins/

# Safe: Skips GUI tests by default
pytest tests/

# Explicit: Run GUI tests only when you want
pytest tests/gui_frontend/ -m gui
```

---

## Benefits of This Architecture

1. **No Circular Dependencies:** Impossible by design
2. **Testable Core:** Physics engine testable without GUI
3. **Clear Responsibilities:** Each package has a single purpose
4. **Easy Refactoring:** Dependencies flow one way
5. **Independent Development:** Teams can work on different levels
6. **CI/CD Friendly:** Core tests run fast without GUI
7. **Reusable Core:** Core engine usable in other contexts

---

## Anti-Patterns to Avoid

### ❌ Don't Create These Files:

- `utils.py` (at root)
- `common.py` (at root)
- `helpers.py` (at root)
- `shared.py` (at root)

**Instead:** Put shared code in `autosim_core` where it belongs.

### ❌ Don't Do This:

```python
# src/autosim_core/engine.py
from gui_frontend import display_results  # ❌ NEVER!

# This violates the hierarchy and creates circular dependencies
```

### ❌ Don't Test Like This:

```bash
# Running all tests without proper markers
pytest -v  # Could accidentally launch GUI
```

**Instead:**
```bash
pytest tests/autosim_core/ -v  # Isolated core tests
```

---

## Enforcement

The architecture is enforced through:

1. **Import guards in tests** - Tests verify no illegal imports
2. **Clear directory structure** - Physical separation of concerns
3. **Documentation** - This file!
4. **Code review** - Check imports during PR review
5. **CI/CD** - Automated checks (future)

---

## Quick Reference

### ✅ ALLOWED Import Patterns

```python
# In autosim_core:
import numpy
import scipy
# NO imports from gui_frontend or vmd_plugins!

# In vmd_plugins:
import autosim_core  # ✅
import matplotlib
# NO imports from gui_frontend!

# In gui_frontend:
import autosim_core  # ✅
import vmd_plugins   # ✅
import customtkinter # ✅
```

### ❌ FORBIDDEN Import Patterns

```python
# In autosim_core:
import gui_frontend        # ❌ NEVER!
import vmd_plugins         # ❌ NEVER!

# In vmd_plugins:
import gui_frontend        # ❌ NEVER!
```

---

## Summary

**Remember the golden rule:**

> Dependencies flow UP the hierarchy, never down.
> 
> Lower levels (core) never know about higher levels (GUI).
> 
> Higher levels (GUI) can use lower levels (core).

This ensures your monorepo stays clean, testable, and maintainable as it grows.
