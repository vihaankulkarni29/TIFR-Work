import MDAnalysis as mda
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

n_atoms = 80000  # Scaled down to kill Python, not the OS
n_frames = 200

print("===========================================")
print("  Synthesizing 'mTOR-Lite' OOM Dataset     ")
print("===========================================")
print(f"Target Atoms : {n_atoms}")
print(f"Target Frames: {n_frames}")
print("Building topology...")

# THE FIX: Explicitly declare 120,000 residues and map 1 atom to each residue
u = mda.Universe.empty(n_atoms, 
                       n_residues=n_atoms, 
                       atom_resindex=np.arange(n_atoms), 
                       trajectory=True)

# Add the dummy names
u.add_TopologyAttr('name', ['CA'] * n_atoms)
u.add_TopologyAttr('resname', ['DUM'] * n_atoms)

# Write the PDB
u.atoms.positions = np.zeros((n_atoms, 3))
u.atoms.write('dummy_mtor.pdb')

print("Writing massive trajectory (do not interrupt)...")
# Write the DCD
with mda.Writer('dummy_mtor.dcd', n_atoms) as W:
    for i in range(n_frames):
        u.atoms.positions = np.random.rand(n_atoms, 3) * 100.0
        W.write(u)

print("Success! The trap is set.")
