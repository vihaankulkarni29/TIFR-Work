"""
Test suite for vmd_plugins package.

These tests can import from autosim_core but NOT from gui_frontend.
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in the path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_vmd_plugins_import():
    """Test that vmd_plugins package can be imported."""
    import vmd_plugins
    assert vmd_plugins.__version__ == "0.1.0"


def test_no_gui_imports():
    """Verify that vmd_plugins does not import GUI packages during testing."""
    import vmd_plugins
    import sys
    
    # Check that GUI packages are not loaded
    assert 'gui_frontend' not in sys.modules, \
        "vmd_plugins should NOT import from gui_frontend"
    assert 'customtkinter' not in sys.modules, \
        "vmd_plugins tests should not trigger GUI imports"


def test_can_import_autosim_core():
    """Verify that vmd_plugins CAN import from autosim_core (allowed dependency)."""
    import autosim_core
    import vmd_plugins
    
    # This is allowed - vmd_plugins can depend on autosim_core
    assert autosim_core.__version__ is not None


@pytest.mark.unit
def test_vmd_interface():
    """Placeholder test for VMD interface."""
    # TODO: Implement VMD interface tests
    assert True, "VMD interface test placeholder"


@pytest.mark.integration
def test_trajectory_analysis():
    """Placeholder test for trajectory analysis."""
    # TODO: Implement trajectory analysis tests
    pass


class TestVisualization:
    """Test suite for visualization utilities."""
    
    def test_plot_generation(self):
        """Test plot generation functionality."""
        # TODO: Implement plot generation tests
        pass
    
    def test_data_processing(self):
        """Test data processing for visualization."""
        # TODO: Implement data processing tests
        pass
