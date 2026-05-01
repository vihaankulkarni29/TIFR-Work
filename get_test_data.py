import shutil
from MDAnalysis.tests.datafiles import PSF, DCD, PDB

# Extracting the built-in benchmark trajectory
shutil.copy(PSF, 'ubiquitin.psf')
shutil.copy(PDB, 'ubiquitin.pdb')
shutil.copy(DCD, 'ubiquitin.dcd')

print("Success: ubiquitin.psf, ubiquitin.pdb, and ubiquitin.dcd have been extracted to your workspace")
