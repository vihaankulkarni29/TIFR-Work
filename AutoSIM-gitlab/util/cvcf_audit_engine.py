import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.transformations as trans
import numpy as np
import os
import sys
import time
try:
    import resource
except ImportError:
    # Fallback to gracefully handle testing on native Windows vs WSL/L-Ubuntu
    resource = None

help_txt='''
Use: python3 cvcf_audit_engine.py <config.txt>
'''

required_keys = {"psffile", "pdbfile", "trjfile"}

if len(sys.argv)!=2 or sys.argv[1] in ["help", "--help", "-help", "-h"]:
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

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        config[key.strip()] = value

if not required_keys.issubset(config):
    print(f"Please check {config_file}.")
    print(f"Mandatory specifications: {required_keys}")
    sys.exit(0)

config["verbose"]=True if config["verbose"].lower() in ["true","1"] else False
config["align"]=True if config["align"].lower() in ["true","1"] else False

try:
    if config["verbose"]:
        print("Loading the trajectory for Deep Telemetry Audit")
        
    u_ref=mda.Universe(os.path.join(cwd,config["psffile"]), os.path.join(cwd,config["pdbfile"]))
    u_traj=mda.Universe(os.path.join(cwd,config["psffile"]), os.path.join(cwd,config["trjfile"]))

    protein = u_traj.select_atoms('protein')
    not_protein = u_traj.select_atoms('not protein')
    
    if config["align"]:
        transforms=[trans.unwrap(protein),
                    trans.center_in_box(protein,wrap=True),
                    trans.wrap(not_protein,compound='residues')]
        u_traj.trajectory.add_transformations(*transforms)

    sel_atoms = u_traj.select_atoms(config["selection"])
    n_atoms = len(sel_atoms)
    
    # Ultra-stream array math initialization
    s1 = np.zeros(n_atoms * 3, dtype='float64')
    s2 = np.zeros(n_atoms * 3, dtype='float64')
    frames = 0
    framelen = len(u_traj.trajectory)

    # RAM output buffer
    output_buffer = []
    output_buffer.append("Frame_Number,CVCF_Value,Total_Elapsed_Time_Sec,Frame_Compute_Time_ms,Peak_RAM_MB\n")
    
    # Start telemetry clock
    global_start_time = time.perf_counter()
    
    for ts in u_traj.trajectory:
        frame_start_time = time.perf_counter()
        frames += 1
        
        if config["align"]:
            align.alignto(u_traj, u_ref, select=config["selection"])
            
        coords = sel_atoms.positions.flatten()
        s1 += coords
        s2 += np.square(coords)
        
        cvcf_val = np.sum((s2 / frames) - np.square(s1 / frames))
        
        frame_end_time = time.perf_counter()
        
        # Telemetry metrics calculations
        total_elapsed_sec = frame_end_time - global_start_time
        frame_compute_ms = (frame_end_time - frame_start_time) * 1000.0
        
        if resource:
            # ru_maxrss is reported in kilobytes on Linux
            peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_ram_mb = peak_ram_kb / 1024.0
        else:
            peak_ram_mb = 0.0
            
        # Write exactly as specified
        output_buffer.append(f"{frames-1},{cvcf_val:.5f},{total_elapsed_sec:.5f},{frame_compute_ms:.5f},{peak_ram_mb:.2f}\n")
        
        if config["verbose"]:
            print(f"Audit Progress: {100 * frames / framelen:.2f} % | RAM: {peak_ram_mb:.2f} MB", end="\r")

    # One-shot Bulk Write to Hard Drive
    trj_name = os.path.splitext(os.path.basename(config["trjfile"]))[0]
    out_filename = f"Master_Telemetry_{trj_name}.csv"
    out_filepath = os.path.join(cwd, out_filename)
    
    with open(out_filepath, "w") as cvcf_out:
        cvcf_out.writelines(output_buffer)

    if config["verbose"]:
        print(f"\nDeep Telemetry Audit completed! Hardware limits mapped safely to: {out_filename}")

except Exception as e:
    print(f"\nError occurred: {e}")
