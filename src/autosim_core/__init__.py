"""
AutoSim Core - Monte Carlo Simulation Engine

This package contains the core computational engine for Monte Carlo simulations
of protein folding dynamics and statistical analysis.

Modules:
    autosim_hpc_orchestrator: High-performance async execution for trajectory analysis
                               and umbrella sampling orchestration
"""

__version__ = "0.1.0"

# Export HPC orchestrator components
try:
    from .autosim_hpc_orchestrator import (
        extract_reaction_coordinates,
        launch_umbrella_window,
        UmbrellaWindowConfig,
        WindowResult,
        PCVectorResult,
        print_window_status_table,
        setup_logging,
    )
    
    __all__ = [
        "extract_reaction_coordinates",
        "launch_umbrella_window",
        "UmbrellaWindowConfig",
        "WindowResult",
        "PCVectorResult",
        "print_window_status_table",
        "setup_logging",
    ]
except ImportError as e:
    # HPC orchestrator requires optional dependencies
    import warnings
    warnings.warn(
        f"HPC orchestrator not available: {e}\n"
        "Install with: pip install -e '.[core]' or pip install MDAnalysis",
        ImportWarning
    )
    __all__ = []
