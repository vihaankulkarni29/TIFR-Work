"""
VMD Remote Control Client via IPC Bridge.

This module provides a Python client to control Visual Molecular Dynamics (VMD)
remotely over a TCP socket connection. It implements affine transformations
for rotating molecular selections while preserving their center of mass.

Key Features:
- TCP socket-based send-only communication (prevents deadlocks)
- Proper 4x4 affine transformation matrices with origin compensation
- scipy/numpy-based rotation composition
- Full TCL command generation and execution

Architecture Notes:
- Socket is strictly send-only to avoid blocking on recv() if VMD Tcl server
  doesn't explicitly echo acknowledgments
- Origin Pendulum Trap mitigation: Always translates to origin before rotation
  and translates back afterward
"""

import socket
import logging
import numpy as np
from typing import List, Literal
from scipy.spatial.transform import Rotation

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - VMDClient - %(levelname)s - %(message)s"
)


class VMDClient:
    """
    Python client for controlling VMD remotely via TCP socket.
    
    Attributes:
        host (str): Hostname or IP address of VMD socket server (default: localhost)
        port (int): Port number of VMD socket server (default: 5555)
        socket (socket.socket | None): TCP socket connection, None if disconnected
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 5.0
    ) -> None:
        """
        Initialize VMD client and establish TCP socket connection.
        
        Args:
            host: Hostname or IP address to connect to (default: "localhost")
            port: Port number to connect to (default: 5555)
            timeout: Socket timeout in seconds (default: 5.0)
            
        Raises:
            ConnectionError: If socket connection fails or times out
        """
        self.host: str = host
        self.port: int = port
        self.socket: socket.socket | None = None
        self.timeout: float = timeout
        
        self._connect()
    
    def _connect(self) -> None:
        """
        Establish TCP socket connection to VMD socket server.
        
        Raises:
            ConnectionError: If connection attempt fails
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to VMD on {self.host}:{self.port}")
        except (socket.timeout, socket.error, OSError) as e:
            self.socket = None
            error_msg = f"Failed to connect to VMD on {self.host}:{self.port}: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    def send_command(self, tcl_string: str) -> None:
        """
        Send a TCL command to VMD via socket.
        
        Encodes the TCL string to UTF-8 bytes and transmits over the socket.
        If socket is disconnected or transmission fails, logs error and optionally
        raises exception.
        
        Args:
            tcl_string: Valid TCL code string to execute in VMD
            
        Raises:
            ConnectionError: If socket is not connected or transmission fails
        """
        if self.socket is None:
            error_msg = "Socket not connected. Call _connect() or re-instantiate VMDClient."
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        
        try:
            # Encode TCL string to UTF-8 bytes and send
            tcl_bytes: bytes = tcl_string.encode('utf-8')
            self.socket.sendall(tcl_bytes)
            logger.debug(f"Sent TCL command: {tcl_string[:80]}...")
        except (socket.error, OSError) as e:
            error_msg = f"Failed to send TCL command: {e}"
            logger.error(error_msg)
            self.socket = None
            raise ConnectionError(error_msg) from e
    
    def rotate_selection(
        self,
        selection_text: str,
        axis: str,
        degrees: float,
        center_of_mass: List[float]
    ) -> None:
        """
        Rotate a VMD atom selection around an arbitrary axis.
        
        This method implements full 4x4 affine transformation with origin
        compensation to avoid the "Origin Pendulum Trap":
        
        1. Translate selection center_of_mass to origin (subtract COM)
        2. Apply rotation matrix around the translated origin
        3. Translate back (add original COM)
        
        The resulting 4x4 matrix is formatted as a TCL list of lists and sent
        to VMD via send_command().
        
        Args:
            selection_text: VMD selection string (e.g., "chain A and resid 1:50")
            axis: Rotation axis as string: "x", "y", or "z" (case-insensitive)
            degrees: Rotation angle in degrees (positive = counter-clockwise)
            center_of_mass: [x, y, z] center of mass in Angstroms
            
        Raises:
            ValueError: If axis is not 'x', 'y', or 'z'
            ConnectionError: From send_command() if socket transmission fails
            
        Example:
            >>> client = VMDClient()
            >>> client.rotate_selection(
            ...     selection_text="chain A",
            ...     axis="z",
            ...     degrees=45.0,
            ...     center_of_mass=[100.5, 200.3, 150.2]
            ... )
        """
        # Validate axis parameter
        axis_lower: str = axis.lower()
        if axis_lower not in ["x", "y", "z"]:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
        
        # Convert degrees to radians
        radians: float = np.radians(degrees)
        
        # Create scipy Rotation object from axis and angle
        # scipy expects axis as array [x, y, z] with unit norm
        axis_vector: np.ndarray = np.zeros(3)
        axis_idx: int = {"x": 0, "y": 1, "z": 2}[axis_lower]
        axis_vector[axis_idx] = 1.0
        rotation: Rotation = Rotation.from_rotvec(axis_vector * radians)
        
        # Get 3x3 rotation matrix from scipy
        rot_matrix_3x3: np.ndarray = rotation.as_matrix()
        
        # Convert center_of_mass to numpy array
        com: np.ndarray = np.array(center_of_mass, dtype=np.float64)
        
        # Construct full 4x4 affine transformation matrix with origin compensation
        # Chain: T_back @ R @ T_to_origin
        # Where T_to_origin translates COM to origin, R rotates, T_back translates back
        
        affine_matrix: np.ndarray = np.eye(4, dtype=np.float64)
        
        # Top-left 3x3: rotation component
        affine_matrix[:3, :3] = rot_matrix_3x3
        
        # Top-right 3x1: translation component
        # Compute: -R @ COM + COM = COM @ (I - R)
        affine_matrix[:3, 3] = com - rot_matrix_3x3 @ com
        
        logger.debug(f"Rotation matrix (3x3):\n{rot_matrix_3x3}")
        logger.debug(f"Affine matrix (4x4):\n{affine_matrix}")
        
        # Format 4x4 matrix as TCL list of lists
        tcl_matrix: str = self._format_tcl_matrix(affine_matrix)
        
        # Generate TCL command
        # atomselect top returns a selection handle
        # $sel move <matrix> applies affine transformation
        # $sel delete cleans up the selection object
        tcl_command: str = (
            f'set sel [atomselect top "{selection_text}"]; '
            f'$sel move {tcl_matrix}; '
            f'$sel delete'
        )
        
        logger.info(
            f"Rotating '{selection_text}' by {degrees}° around {axis_lower}-axis "
            f"with COM at {com}"
        )
        
        # Send command via socket (send-only, no recv() blocking)
        self.send_command(tcl_command)
    
    @staticmethod
    def _format_tcl_matrix(matrix: np.ndarray) -> str:
        """
        Format a 4x4 numpy array as a TCL list of lists.
        
        Converts a 4x4 numpy array into TCL syntax:
        { {m00 m01 m02 m03} {m10 m11 m12 m13} {m20 m21 m22 m23} {m30 m31 m32 m33} }
        
        Args:
            matrix: 4x4 numpy array (float64)
            
        Returns:
            TCL-formatted string representing the matrix as nested lists
            
        Example:
            >>> mat = np.eye(4)
            >>> VMDClient._format_tcl_matrix(mat)
            '{ {1.0 0.0 0.0 0.0} {0.0 1.0 0.0 0.0} {0.0 0.0 1.0 0.0} {0.0 0.0 0.0 1.0} }'
        """
        rows: List[str] = []
        for i in range(4):
            row_values: List[str] = [f"{matrix[i, j]:.10g}" for j in range(4)]
            row_tcl: str = "{" + " ".join(row_values) + "}"
            rows.append(row_tcl)
        
        tcl_matrix: str = "{ " + " ".join(rows) + " }"
        return tcl_matrix
    
    def disconnect(self) -> None:
        """
        Close the TCP socket connection to VMD.
        
        Safe to call multiple times; subsequent calls are no-ops if already disconnected.
        """
        if self.socket is not None:
            try:
                self.socket.close()
                logger.info(f"Disconnected from VMD on {self.host}:{self.port}")
            except socket.error as e:
                logger.warning(f"Error closing socket: {e}")
            finally:
                self.socket = None
    
    def __enter__(self) -> "VMDClient":
        """Context manager entry point."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point; ensures socket cleanup."""
        self.disconnect()
