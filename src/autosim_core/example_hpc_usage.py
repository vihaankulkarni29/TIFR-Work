"""
Example usage and testing script for the AutoSIM HPC Orchestrator.

This script demonstrates how to use the vectorized PCA extraction and
async umbrella spawning functionality.

Usage:
    python -m autosim_core.example_hpc_usage
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autosim_core.autosim_hpc_orchestrator import (
    extract_reaction_coordinates,
    launch_umbrella_window,
    UmbrellaWindowConfig,
    print_window_status_table,
    logger,
    DEFAULT_TOPOLOGY,
    DEFAULT_TRAJECTORY,
)

try:
    import MDAnalysis as mda
except ImportError:
    print("Error: MDAnalysis not installed. Install with: pip install MDAnalysis")
    sys.exit(1)


# ============================================================================
# Example 1: Vectorized PCA Extraction
# ============================================================================

def example_pca_extraction():
    """
    Demonstrate vectorized PCA extraction on a test trajectory.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Vectorized PCA Extraction (Run 1)")
    print("="*80 + "\n")
    
    # Check if test data exists
    if not DEFAULT_TOPOLOGY.exists() or not DEFAULT_TRAJECTORY.exists():
        print(f"⚠️  Test data not found at:")
        print(f"   Topology:   {DEFAULT_TOPOLOGY}")
        print(f"   Trajectory: {DEFAULT_TRAJECTORY}")
        print("\nTo create test data structure:")
        print("   mkdir -p autosim_workspace/data")
        print("   # Copy your .tpr and .xtc files there")
        print("\nFor now, creating a mock demonstration...")
        
        # Create a synthetic trajectory for demonstration
        return _create_synthetic_pca_demo()
    
    # Load trajectory
    print(f"Loading trajectory from {DEFAULT_TRAJECTORY}...")
    u = mda.Universe(str(DEFAULT_TOPOLOGY), str(DEFAULT_TRAJECTORY))
    
    print(f"✓ Loaded {u.trajectory.n_frames} frames with {len(u.atoms)} atoms")
    
    # Extract PCA
    print("\nExtracting principal components (vectorized)...")
    result = extract_reaction_coordinates(
        u,
        atom_selection="protein and name CA",  # C-alpha atoms
        memory_efficient=False  # Use in-memory for speed
    )
    
    # Display results
    print("\n" + "-"*80)
    print("PCA RESULTS")
    print("-"*80)
    print(f"Frames analyzed:      {result.n_frames}")
    print(f"Atoms selected:       {result.n_atoms}")
    print(f"Features (dims):      {len(result.pc1_eigenvector)}")
    print(f"PC1 eigenvalue:       {result.eigenvalue:.4e}")
    print(f"Variance explained:   {result.variance_explained:.2%}")
    print(f"Computation time:     {result.computation_time_ms:.1f} ms")
    print("-"*80)
    
    print(f"\n✓ PC1 eigenvector shape: {result.pc1_eigenvector.shape}")
    print(f"  First 10 components: {result.pc1_eigenvector[:10]}")
    
    return result


def _create_synthetic_pca_demo():
    """
    Create a synthetic trajectory for demonstration purposes.
    """
    print("\nCreating synthetic trajectory for demonstration...")
    
    # Create a simple synthetic trajectory with known structure
    n_frames = 100
    n_atoms = 50
    n_features = n_atoms * 3
    
    # Generate correlated random motion along a principal direction
    np.random.seed(42)
    
    # Create a random principal direction
    true_pc = np.random.randn(n_features)
    true_pc /= np.linalg.norm(true_pc)
    
    # Generate trajectory with motion along this PC
    coordinates = np.zeros((n_frames, n_features))
    for i in range(n_frames):
        # Motion along PC1 + some noise
        t = i / n_frames
        coordinates[i] = true_pc * np.sin(2 * np.pi * t) + np.random.randn(n_features) * 0.1
    
    # Mock Universe (simplified - actual implementation would use MDAnalysis)
    print(f"✓ Created synthetic trajectory: {n_frames} frames, {n_atoms} atoms")
    
    # Perform PCA manually for demo
    mean_coords = np.mean(coordinates, axis=0)
    centered = coordinates - mean_coords
    cov_matrix = (centered.T @ centered) / (n_frames - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    pc1 = eigenvectors[:, 0]
    variance = eigenvalues[0] / np.sum(eigenvalues)
    
    print("\n" + "-"*80)
    print("SYNTHETIC PCA RESULTS (Demo)")
    print("-"*80)
    print(f"Frames analyzed:      {n_frames}")
    print(f"Atoms selected:       {n_atoms}")
    print(f"Features (dims):      {n_features}")
    print(f"PC1 eigenvalue:       {eigenvalues[0]:.4e}")
    print(f"Variance explained:   {variance:.2%}")
    print("-"*80)
    
    return pc1


# ============================================================================
# Example 2: Async Umbrella Window Launch
# ============================================================================

async def example_umbrella_window():
    """
    Demonstrate async umbrella window launching.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Async Umbrella Window Launch (Run 2)")
    print("="*80 + "\n")
    
    # Generate a mock PC vector
    print("Creating mock PC vector for demonstration...")
    n_atoms = 50
    pc_vector = np.random.randn(n_atoms * 3)
    pc_vector /= np.linalg.norm(pc_vector)
    
    print(f"✓ PC vector created: {len(pc_vector)} features\n")
    
    # Check for GROMACS
    print("Checking for GROMACS installation...")
    gromacs_available = await _check_gromacs()
    
    if not gromacs_available:
        print("⚠️  GROMACS not found. Window launch will be simulated.\n")
        print("To install GROMACS:")
        print("   - Ubuntu/Debian: sudo apt-get install gromacs")
        print("   - Conda: conda install -c conda-forge gromacs")
        print("   - From source: http://manual.gromacs.org/documentation/")
        
        # Simulate window launch
        return await _simulate_umbrella_window(pc_vector)
    
    # Create configuration for umbrella window
    config = UmbrellaWindowConfig(
        window_id=0,
        pc_vector=pc_vector,
        force_constant=1000.0,  # kJ/(mol·nm²)
        center_position=0.5,     # nm
        nsteps=5000,            # Short run for demo
        stride=10
    )
    
    # Check for topology file
    if not DEFAULT_TOPOLOGY.exists():
        print(f"⚠️  Topology file not found: {DEFAULT_TOPOLOGY}")
        print("    Window launch will be simulated.\n")
        return await _simulate_umbrella_window(pc_vector)
    
    # Launch window
    output_dir = Path("autosim_workspace/umbrella_output")
    
    print(f"Launching umbrella window {config.window_id}...")
    print(f"  Force constant: {config.force_constant} kJ/(mol·nm²)")
    print(f"  Center:         {config.center_position} nm")
    print(f"  Steps:          {config.nsteps}")
    print(f"  Output dir:     {output_dir}\n")
    
    result = await launch_umbrella_window(
        config=config,
        topology_file=DEFAULT_TOPOLOGY,
        output_dir=output_dir,
        num_threads=4,
        num_openmp_threads=1
    )
    
    # Display results
    print("\n" + "-"*80)
    print("UMBRELLA WINDOW RESULT")
    print("-"*80)
    print(f"Window ID:      {result.window_id}")
    print(f"Status:         {result.status}")
    print(f"Exit code:      {result.exit_code}")
    print(f"Runtime:        {result.runtime_seconds:.2f} s")
    
    if result.ns_per_day is not None:
        print(f"Performance:    {result.ns_per_day:.2f} ns/day")
    
    if result.final_potential_energy is not None:
        print(f"Final energy:   {result.final_potential_energy:.2f} kJ/mol")
    
    print("-"*80)
    
    # Show table format
    print_window_status_table([result])
    
    return result


async def _check_gromacs() -> bool:
    """Check if GROMACS is available."""
    try:
        process = await asyncio.create_subprocess_exec(
            "gmx", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Extract version
            version_line = stdout.decode().split('\n')[0]
            print(f"✓ GROMACS found: {version_line}\n")
            return True
        else:
            return False
            
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking GROMACS: {e}")
        return False


async def _simulate_umbrella_window(pc_vector: np.ndarray):
    """Simulate umbrella window launch for demonstration."""
    from autosim_core.autosim_hpc_orchestrator import WindowResult
    
    print("Simulating umbrella window launch...")
    
    # Simulate some work
    await asyncio.sleep(0.5)
    
    # Create mock result
    result = WindowResult(
        window_id=0,
        status="Completed (Simulated)",
        exit_code=0,
        ns_per_day=12.5,
        final_potential_energy=-123456.78,
        runtime_seconds=0.5,
        stdout="(Simulated GROMACS output)",
        stderr=""
    )
    
    print("\n" + "-"*80)
    print("SIMULATED UMBRELLA WINDOW RESULT")
    print("-"*80)
    print(f"Window ID:      {result.window_id}")
    print(f"Status:         {result.status}")
    print(f"Performance:    {result.ns_per_day:.2f} ns/day")
    print(f"Final energy:   {result.final_potential_energy:.2f} kJ/mol")
    print("-"*80)
    
    print_window_status_table([result])
    
    return result


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AutoSIM HPC Orchestrator - Example Usage & Testing")
    print("="*80)
    
    # Example 1: PCA
    print("\n[1/2] Running PCA extraction example...")
    pc_vector = example_pca_extraction()
    
    # Example 2: Umbrella window
    print("\n[2/2] Running umbrella window example...")
    asyncio.run(example_umbrella_window())
    
    print("\n" + "="*80)
    print("✓ All examples completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the generated autosim_compute.log for detailed execution logs")
    print("  2. Implement Run 3 (Orchestrator Loop) for batch execution")
    print("  3. Add your real trajectory data to autosim_workspace/data/")
    print("  4. Customize Plumed restraints for your specific system")
    print("\n")


if __name__ == "__main__":
    main()
