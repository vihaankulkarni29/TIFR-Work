"""
VMD Plugins - VMD Integration and Visualization

This package provides interfaces and plugins for VMD (Visual Molecular Dynamics)
integration, trajectory analysis, and molecular visualization utilities.

Key Components:
- VMDClient: IPC bridge client for remote VMD control via TCP socket
  with affine transformation support for molecular rotations
"""

from vmd_plugins.vmd_commander import VMDClient

__version__ = "0.1.0"
__all__ = ["VMDClient"]
