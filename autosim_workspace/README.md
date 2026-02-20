# AutoSIM Workspace

This directory contains data files and outputs for AutoSIM HPC orchestrator.

## Directory Structure

```
autosim_workspace/
├── data/                    # Input data (trajectories, topologies)
│   ├── system.tpr          # GROMACS topology/structure file
│   └── trajectory.xtc      # Trajectory file
├── umbrella_output/        # Umbrella sampling outputs
│   ├── window_0000/
│   ├── window_0001/
│   └── ...
└── logs/                   # Computation logs
```

## Setting Up Test Data

### Option 1: Use Your Own Data

Copy your GROMACS files:
```bash
mkdir -p autosim_workspace/data
cp /path/to/your/system.tpr autosim_workspace/data/
cp /path/to/your/trajectory.xtc autosim_workspace/data/
```

### Option 2: Download Test System

For testing purposes, you can use a small protein system:

```bash
# Download a test system (e.g., from MDAnalysis test data)
python << EOF
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD

# Download test files
import shutil
import os

os.makedirs('autosim_workspace/data', exist_ok=True)
shutil.copy(PSF, 'autosim_workspace/data/system.psf')
shutil.copy(DCD, 'autosim_workspace/data/trajectory.dcd')

print("✓ Test data downloaded to autosim_workspace/data/")
EOF
```

### Option 3: Generate Synthetic Data

Use the example script which automatically generates synthetic data for testing:

```bash
python -m autosim_core.example_hpc_usage
```

## File Formats

### Topology Files
- `.tpr` - GROMACS topology (contains structure, topology, and run parameters)
- `.psf` - CHARMM/NAMD topology
- `.pdb` - Protein Data Bank format (structure only)

### Trajectory Files
- `.xtc` - GROMACS compressed trajectory (recommended)
- `.trr` - GROMACS full-precision trajectory
- `.dcd` - CHARMM/NAMD trajectory

## Running the Orchestrator

Once data is in place:

```bash
# Run examples
python -m autosim_core.example_hpc_usage

# Or use in your own script
python << EOF
import MDAnalysis as mda
from autosim_core.autosim_hpc_orchestrator import extract_reaction_coordinates

u = mda.Universe(
    'autosim_workspace/data/system.tpr',
    'autosim_workspace/data/trajectory.xtc'
)

result = extract_reaction_coordinates(u, 'protein and name CA')
print(f"PC1 explains {result.variance_explained:.2%} of variance")
EOF
```

## Output Files

### Umbrella Sampling Output

Each window creates:
```
umbrella_output/window_0000/
├── plumed.dat                 # Plumed restraint configuration
├── umbrella_0000.log          # GROMACS log file
├── umbrella_0000.xtc          # Output trajectory
├── umbrella_0000.edr          # Energy file
└── COLVAR                     # Plumed collective variables
```

### Logs

- `autosim_compute.log` - Detailed timestamped logs (in project root)

## Disk Space Requirements

Estimated space per umbrella window:
- Small system (1000 atoms, 50k steps): ~10-50 MB
- Medium system (10k atoms, 500k steps): ~500 MB - 2 GB
- Large system (100k atoms, 5M steps): ~10-50 GB

Plan accordingly!

## Cleaning Up

To remove output files:

```bash
# Remove umbrella outputs
rm -rf autosim_workspace/umbrella_output/*

# Remove logs
rm autosim_compute.log

# Keep data files
# (!)  Do NOT run: rm -rf autosim_workspace/data
```

## Troubleshooting

### "Universe not found" error

Make sure files exist:
```bash
ls -lh autosim_workspace/data/
```

### "MDAnalysis cannot read file" error

Check file format compatibility:
```python
import MDAnalysis as mda
u = mda.Universe('autosim_workspace/data/system.tpr')  # Test topology
print(f"Loaded {len(u.atoms)} atoms")
```

### GROMACS simulation fails

Check GROMACS installation:
```bash
gmx --version
```

Check threading parameters in launch_umbrella_window() call.

## Next Steps

1. ✓ Set up data in `autosim_workspace/data/`
2. ✓ Run PCA extraction on your trajectory
3. ✓ Launch test umbrella windows
4. ⏳ Implement Run 3 (batch orchestrator)
5. ⏳ Analyze PMF (Potential of Mean Force)

Happy computing! 🚀
