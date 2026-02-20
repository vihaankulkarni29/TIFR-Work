"""
Test suite for autosim_core package.

These tests are STRICTLY ISOLATED from GUI dependencies.
They should NEVER import from gui_frontend or vmd_plugins.
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in the path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_autosim_core_import():
    """Test that autosim_core package can be imported without GUI dependencies."""
    import autosim_core
    assert autosim_core.__version__ == "0.1.0"


def test_autosim_core_version():
    """Test that autosim_core has correct version."""
    import autosim_core
    assert hasattr(autosim_core, "__version__")
    assert isinstance(autosim_core.__version__, str)


def test_no_gui_imports():
    """Verify that autosim_core does not import GUI packages."""
    import autosim_core
    import sys
    
    # Check that GUI packages are not loaded by importing core
    assert 'gui_frontend' not in sys.modules, \
        "autosim_core should NEVER import from gui_frontend"
    assert 'customtkinter' not in sys.modules, \
        "autosim_core tests should not trigger GUI imports"


@pytest.mark.unit
def test_placeholder_simulation():
    """Placeholder test for simulation functionality."""
    # TODO: Implement actual simulation tests
    assert True, "Simulation test placeholder"


@pytest.mark.slow
def test_monte_carlo_convergence():
    """Placeholder for Monte Carlo convergence test."""
    # TODO: Implement convergence testing
    # This will test that MC simulations converge to expected distributions
    pass


class TestSimulationEngine:
    """Test suite for the core simulation engine."""
    
    def test_initialization(self):
        """Test simulation engine initialization."""
        # TODO: Implement initialization tests
        pass
    
    def test_parameter_validation(self):
        """Test parameter validation in simulation setup."""
        # TODO: Implement parameter validation tests
        pass
    
    def test_computational_correctness(self):
        """Test that physics computations are correct."""
        # TODO: Implement physics correctness tests
        pass
