import MDAnalysis as mda
import numpy as np
import warnings

# Suppress all warnings about empty universes or missing crystallographic data
warnings.filterwarnings("ignore")

n_atoms = 120000
n_frames = 200

print("===================================================")
print("  Synthesizing 'mTOR-Lite' OOM Stress Dataset      ")
print("===================================================")
print(f"Target Atoms : {n_atoms}")
print(f"Target Frames: {n_frames}")
print("Building topology...")

# Create empty universe with trajectory support
u = mda.Universe.empty(n_atoms, trajectory=True)

# Add dummy 'CA' names and basic residue data
u.add_TopologyAttr('name', ['CA'] * n_atoms)
u.add_TopologyAttr('resname', ['DUM'] * n_atoms)
u.add_TopologyAttr('resid', list(range(1, n_atoms + 1)))

# Initialize with zero coordinates for the PDB
u.atoms.positions = np.zeros((n_atoms, 3))
u.atoms.write("dummy_mtor.pdb")
print("Topology written to dummy_mtor.pdb")

print("Generating random trajectory frames and writing to disk...")
with mda.Writer("dummy_mtor.dcd", n_atoms=n_atoms) as W:
    for i in range(n_frames):
        # Generate random coordinates for the frame
        u.atoms.positions = np.random.rand(n_atoms, 3) * 100.0
        W.write(u)
        
        if (i + 1) % 20 == 0:
            print(f"  -> Wrote {i + 1}/{n_frames} frames")

print("===================================================")
print("  Synthesis Complete!                              ")
print("===================================================")
