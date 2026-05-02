import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.transformations as trans
import numpy as np
import os
import sys
import time
import argparse

# Memory profiling setup supporting both Windows and Unix
try:
    import psutil
    def get_memory_mb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
except ImportError:
    try:
        import resource
        def get_memory_mb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        def get_memory_mb():
            return 0.0

parser = argparse.ArgumentParser(description="Universal CVCF Telemetry Engine")
parser.add_argument("config", help="Path to configuration file")
parser.add_argument("--script-type", required=True, choices=['Legacy', 'Ultra'], help="Execution mode")
parser.add_argument("--protein-name", required=True, help="Target molecule label")
args = parser.parse_args()

cwd = os.getcwd()
config_file = os.path.join(cwd, args.config)
script_type = args.script_type
protein_name = args.protein_name

if not (os.path.exists(config_file) and os.path.isfile(config_file)):
    print(f"Config file {config_file} missing.")
    sys.exit(1)

config = {'selection':'name CA','verbose':'false','align':'false'}
with open(config_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip().strip('"').strip("'")

config["verbose"] = True if config["verbose"].lower() in ["true","1"] else False
config["align"] = True if config["align"].lower() in ["true","1"] else False

try:
    u_ref = mda.Universe(os.path.join(cwd,config["psffile"]), os.path.join(cwd,config["pdbfile"]))
    u_traj = mda.Universe(os.path.join(cwd,config["psffile"]), os.path.join(cwd,config["trjfile"]))
    
    if config["align"]:
        protein = u_traj.select_atoms('protein')
        not_protein = u_traj.select_atoms('not protein')
        transforms = [trans.unwrap(protein), trans.center_in_box(protein,wrap=True), trans.wrap(not_protein,compound='residues')]
        u_traj.trajectory.add_transformations(*transforms)

    sel_atoms = u_traj.select_atoms(config["selection"])
    n_atoms = len(sel_atoms)
    
    s1 = np.zeros(n_atoms * 3, dtype='float64')
    s2 = np.zeros(n_atoms * 3, dtype='float64')
    frames = 0
    
    output_buffer = []
    all_frames = [] # Global memory leak sink for Legacy mode
    norm = 100000.0

    global_start_time = time.perf_counter()
    
    for ts in u_traj.trajectory:
        frame_start_time = time.perf_counter()
        frames += 1
        
        if config["align"]:
            align.alignto(u_traj, u_ref, select=config["selection"])
            
        coords = sel_atoms.positions.flatten()
        
        # Dual-Mode Architecture Dispatch
        if script_type == 'Legacy':
            all_frames.append(np.copy(coords)) # Induce O(N) memory growth footprint
            coords_norm = coords / norm
            s1 += coords_norm
            s2 += np.multiply(coords_norm, coords_norm)
            cvcf_val = (norm * norm) * np.sum((s2 / frames) - np.square(s1 / frames))
        else: # Ultra mode
            s1 += coords
            s2 += np.square(coords)
            cvcf_val = np.sum((s2 / frames) - np.square(s1 / frames))
            
        frame_end_time = time.perf_counter()
        
        total_elapsed_sec = frame_end_time - global_start_time
        frame_compute_ms = (frame_end_time - frame_start_time) * 1000.0
        mem_mb = get_memory_mb()
        
        output_buffer.append(f"{protein_name},{script_type},{frames-1},{cvcf_val:.5f},{total_elapsed_sec:.5f},{frame_compute_ms:.5f},{mem_mb:.2f}\n")

    # Bulk Write Appended Mode
    with open("telemetry_data.csv", "a") as f_out:
        f_out.writelines(output_buffer)

except Exception as e:
    print(f"\nEngine crashed during execution: {e}")
    sys.exit(1)
