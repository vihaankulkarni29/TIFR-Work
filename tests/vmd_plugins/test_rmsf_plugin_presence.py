"""Structural tests for the Vihaan RMSF VMD Tcl plugin.

These tests do not execute VMD or Tcl runtime code. They validate that the
plugin file is present and contains required integration hooks and safeguards.
"""

from pathlib import Path


def test_rmsf_plugin_file_exists() -> None:
    """Plugin file should exist under vmd_plugins package directory."""
    plugin_path = Path("src/vmd_plugins/vihaan_rmsf_plugin.tcl")
    assert plugin_path.exists(), "Expected RMSF plugin file is missing"


def test_rmsf_plugin_contains_required_constructs() -> None:
    """Ensure required namespace, GUI entry points, and menu registration exist."""
    plugin_path = Path("src/vmd_plugins/vihaan_rmsf_plugin.tcl")
    content = plugin_path.read_text(encoding="utf-8")

    required_snippets = [
        "namespace eval ::VihaanRMSF::",
        "proc gui {}",
        "variable calc_bfactor 0",
        "Signature: Convert to Theoretical B-factor",
        "proc calculate {}",
        "$sel frame $i",
        "$sel update",
        "$sel set user $values",
        "$sel delete",
        "menu tk register \"Vihaan RMSF Calculator\" ::VihaanRMSF::gui",
    ]

    for snippet in required_snippets:
        assert snippet in content, f"Missing required plugin construct: {snippet}"


def test_rmsf_plugin_bfactor_formula_present() -> None:
    """Verify theoretical B-factor conversion formula is implemented."""
    plugin_path = Path("src/vmd_plugins/vihaan_rmsf_plugin.tcl")
    content = plugin_path.read_text(encoding="utf-8")

    # Accept explicit pi expansion implemented with acos(-1.0)
    assert "(8.0 * acos(-1.0) * acos(-1.0)) / 3.0" in content
    assert "set outval [expr {$bscale * $rmsf * $rmsf}]" in content
