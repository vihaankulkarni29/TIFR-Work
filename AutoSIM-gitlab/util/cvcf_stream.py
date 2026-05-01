import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.transformations as trans
import numpy as np
import os
import sys

help_txt='''
Use: python3 cvcf_stream.py <config.txt>

<config.txt> : Replace with config file.

Contents of typical <config.txt> file:

psffile="./test.psf"        : PSF file to read PDB and trajectory
pdbfile="./test.pdb"        : PDB file to be used as reference
trjfile="./test.xtc"        : Trajectory file for CVCF computation (DCD or XTC)
outfile="./cvcf.txt"        : Output file (text format) for CVCF
selection="name CA"         : Selection of atoms for fitting and CVCF
                              optional, default: CA atoms (name CA)
align=true/false/1/0        : Align the trajectory (in-memory)
                              optional, default: false (do not align)
verbose=true/false/1/0      : Show detailed progress of computations
                              optional, default: false (do not show)

To preview this help:
python3 cvcf_stream.py <help/-h/-help/--help>

'''

required_keys = {"psffile", "pdbfile", "trjfile", "outfile"}

if len(sys.argv)!=2 or sys.argv[1]=="help" or sys.argv[1]=="--help" or sys.argv[1]=="-help" or sys.argv[1]=="-h":
    print(help_txt)
    sys.exit(0)

cwd=os.getcwd()
config_file=os.path.join(cwd,sys.argv[1])

if not (os.path.exists(config_file) and os.path.isfile(config_file)):
    print(f"The file {config_file} does not exist.\n{help_txt}")
    sys.exit(0)


config = {'selection':'name CA','verbose':'false','align':'false'}

with open(config_file, "r") as f:
    for line in f:
        line = line.strip()

        # Skip empty lines or comments
        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)

        # Remove quotes
        value = value.strip().strip('"').strip("'")

        config[key.strip()] = value

if not required_keys.issubset(config):
    print(f"Please check {config_file}.")
    print(f"Mandatory specifications: {required_keys}")
    print()
    print(help_txt)
    sys.exit(0)

config["verbose"]=True if config["verbose"].lower() in ["true","1"] else False
config["align"]=True if config["align"].lower() in ["true","1"] else False

try:

    if config["verbose"]:
        print("Loading the trajectory")
        
    # Read the trajectory
    u_ref=mda.Universe(     os.path.join(cwd,config["psffile"]),    os.path.join(cwd,config["pdbfile"])     )
    u_traj=mda.Universe(    os.path.join(cwd,config["psffile"]),    os.path.join(cwd,config["trjfile"])     )

    # Unwrap the protein in the box
    protein = u_traj.select_atoms('protein')
    not_protein = u_traj.select_atoms('not protein')
    
    if config["verbose"]:
        print("Trajectory load complete")
    
    if config["align"]:
        if config["verbose"]:
            print("Configuring trajectory transformations for unwrapping...")
        transforms=[trans.unwrap(protein),
                    trans.center_in_box(protein,wrap=True),
                    trans.wrap(not_protein,compound='residues')]
        u_traj.trajectory.add_transformations(*transforms)
        
        if config["verbose"]:
            print("Per-frame alignment configured.")

    # Get the selection AtomGroup whose coordinates we will compute CVCF on
    sel_atoms = u_traj.select_atoms(config["selection"])
    
    # Calculate dimensions for the flattened coordinates
    n_atoms = len(sel_atoms)
    
    # Use the fast algorithm for CVCF computations
    s1 = np.zeros(n_atoms * 3, dtype='float64')
    s2 = np.zeros(n_atoms * 3, dtype='float64')
    frames = 0
    norm = 100000.0

    framelen = len(u_traj.trajectory)

    if config["verbose"]:
        print("Computing CVCF in a memory-safe stream")
    
    # Open the output file prior to the loop to stream data frame-by-frame
    cvcf_out = open(os.path.join(cwd, config["outfile"]), "w")
    
    # CVCF calculation loop - generator reads strictly one frame into RAM at a time
    for ts in u_traj.trajectory:
        frames += 1
        
        # In-loop alignment to avoid loading the entire trajectory into memory via align.AlignTraj
        if config["align"]:
            align.alignto(u_traj, u_ref, select=config["selection"])
            
        # Extract the flattened coordinates for the current frame
        coords = sel_atoms.positions.flatten()
        
        coords_norm = coords / norm
        s1 += coords_norm
        s2 += np.multiply(coords_norm, coords_norm)
        
        # Calculate cvcf_val exactly as per the previous logic
        cvcf_val = (norm * norm) * np.sum((s2 / frames) - np.square(s1 / frames))
        
        # Write to disk immediately
        cvcf_out.write(f"{(frames - 1):10.0f}{cvcf_val:20.5f}\n")
        
        if config["verbose"]:
            print("CVCF Progress: ", (100 * frames / framelen), "%", end="\r")

    cvcf_out.close()

    if config["verbose"]:
        print()
        print("CVCF computed and streamed to disk successfully")

except Exception as e:
    print()
    print("Error occurred:", e)
    print(help_txt)
