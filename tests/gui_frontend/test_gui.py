"""
Test suite for gui_frontend package.

GUI tests are isolated and should NOT run during core/plugin testing.
These tests can import from autosim_core and vmd_plugins.
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in the path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_gui_frontend_import():
    """Test that gui_frontend package can be imported."""
    import gui_frontend
    assert gui_frontend.__version__ == "0.1.0"


@pytest.mark.unit
def test_main_module_exists():
    """Test that main module exists and can be imported."""
    from gui_frontend import main
    assert hasattr(main, "main")
    assert callable(main.main)


def test_can_import_dependencies():
    """Verify that gui_frontend CAN import from autosim_core and vmd_plugins."""
    import autosim_core
    import vmd_plugins
    import gui_frontend
    
    # This is allowed - gui_frontend can depend on both
    assert autosim_core.__version__ is not None
    assert vmd_plugins.__version__ is not None


# Note: GUI tests require special handling and should be isolated
@pytest.mark.gui
@pytest.mark.skip(reason="GUI testing requires display environment and should be isolated from core tests")
def test_gui_initialization():
    """Placeholder for GUI initialization test."""
    # TODO: Implement GUI tests with proper mocking or testing framework
    # Use pytest-qt or similar for actual GUI testing
    pass


class TestGUIComponents:
    """Test suite for GUI components - ALL ISOLATED FROM CORE TESTS."""
    
    @pytest.mark.gui
    @pytest.mark.skip(reason="GUI testing requires display environment")
    def test_window_creation(self):
        """Test main window creation."""
        # TODO: Implement window creation tests
        pass
    
    @pytest.mark.gui
    @pytest.mark.skip(reason="GUI testing requires display environment")
    def test_event_handling(self):
        """Test GUI event handling."""
        # TODO: Implement event handling tests
        pass
