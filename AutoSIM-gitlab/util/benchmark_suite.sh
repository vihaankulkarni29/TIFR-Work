#!/bin/bash

echo "======================================================="
echo "  AutoSIM DevOps Pipeline: Deep Telemetry Audit Suite  "
echo "======================================================="

# --- 1. Dynamic Config Generation ---
echo "Generating dynamic configuration files..."

cat <<EOF > config_small.txt
psffile="ubiquitin.psf"
pdbfile="ubiquitin.pdb"
trjfile="ubiquitin.dcd"
outfile="cvcf_output.txt"
selection="name CA"
align=false
verbose=false
EOF

cat <<EOF > config_massive.txt
psffile="dummy_mtor.pdb"
pdbfile="dummy_mtor.pdb"
trjfile="dummy_mtor.dcd"
outfile="cvcf_output.txt"
selection="name CA"
align=false
verbose=false
EOF

# --- 2. Telemetry Extraction Function ---
extract_telemetry() {
    local log=$1
    local out=$2
    local exit_code=$3
    local stage=$4
    local script=$5
    local target=$6
    local parity=$7

    if [ "$exit_code" -eq 0 ]; then
        status="Success"
    elif [ "$exit_code" -eq 137 ]; then
        status="OOM Kill (137)"
    else
        status="Crash ($exit_code)"
    fi

    # Extract RAM (kbytes to MB)
    if grep -q "Maximum resident set size" "$log"; then
        peak_kb=$(grep "Maximum resident set size" "$log" | awk '{print $6}')
        peak_mb=$(echo "scale=2; $peak_kb / 1024" | bc 2>/dev/null || echo "$((peak_kb / 1024))")
    else
        peak_mb="N/A"
    fi

    # Extract CPU Time
    if grep -q "User time" "$log"; then
        user_time=$(grep "User time (seconds)" "$log" | awk '{print $4}')
        sys_time=$(grep "System time (seconds)" "$log" | awk '{print $4}')
        cpu_time=$(echo "$user_time + $sys_time" | bc 2>/dev/null || awk "BEGIN {print $user_time + $sys_time}")
    else
        cpu_time="N/A"
    fi

    # Extract Disk IO
    if grep -q "File system inputs" "$log"; then
        disk_io=$(grep "File system inputs" "$log" | awk '{print $5}')
    else
        disk_io="N/A"
    fi

    # Extract Output Size
    if [ -f "$out" ]; then
        out_size=$(ls -lh "$out" | awk '{print $5}')
    else
        out_size="Missing"
    fi

    # Write to CSV
    echo "${stage},${script},${target},${peak_mb},${cpu_time},${disk_io},${out_size},${status},${parity}" >> AutoSIM_Audit_Report.csv
}

# Initialize CSV
echo "Stage,Script,Target_Molecule,Peak_RAM_MB,CPU_Time_Sec,Disk_IO_Ops,Output_Size,Exit_Status,Math_Parity" > AutoSIM_Audit_Report.csv

# Cleanup old logs
rm -f profiler_*.log cvcf_*.txt cvcf_output.txt diff_parity.txt

# --- STAGE 1: Control on Ubiquitin ---
echo -e "\n[Stage 1/4] Executing Control (cvcf.py) on Ubiquitin..."
/usr/bin/time -v python3 cvcf.py config_small.txt 2> profiler_ctrl_ubi.log
exit_1=$?
if [ -f "cvcf_output.txt" ]; then mv cvcf_output.txt cvcf_ctrl_ubi.txt; fi
extract_telemetry profiler_ctrl_ubi.log cvcf_ctrl_ubi.txt $exit_1 1 "cvcf.py" "Ubiquitin" "Baseline"

# --- STAGE 2: Refactor on Ubiquitin ---
echo -e "\n[Stage 2/4] Executing Refactor (cvcf_stream.py) on Ubiquitin..."
/usr/bin/time -v python3 cvcf_stream.py config_small.txt 2> profiler_ref_ubi.log
exit_2=$?
if [ -f "cvcf_output.txt" ]; then mv cvcf_output.txt cvcf_ref_ubi.txt; fi

# Mathematical Parity Test
parity_status="N/A"
if [ -f "cvcf_ctrl_ubi.txt" ] && [ -f "cvcf_ref_ubi.txt" ]; then
    diff cvcf_ctrl_ubi.txt cvcf_ref_ubi.txt > diff_parity.txt
    if [ -s diff_parity.txt ]; then
        parity_status="FAILED"
    else
        parity_status="100% PARITY"
    fi
fi
extract_telemetry profiler_ref_ubi.log cvcf_ref_ubi.txt $exit_2 2 "cvcf_stream.py" "Ubiquitin" "$parity_status"

# --- STAGE 3: Control on mTOR-Lite ---
echo -e "\n[Stage 3/4] Executing Control (cvcf.py) on mTOR-Lite... (Expect OOM Crash)"
/usr/bin/time -v python3 cvcf.py config_massive.txt 2> profiler_ctrl_mtor.log
exit_3=$?
if [ -f "cvcf_output.txt" ]; then mv cvcf_output.txt cvcf_ctrl_mtor.txt; fi
extract_telemetry profiler_ctrl_mtor.log cvcf_ctrl_mtor.txt $exit_3 3 "cvcf.py" "mTOR-Lite" "N/A"

# --- STAGE 4: Refactor on mTOR-Lite ---
echo -e "\n[Stage 4/4] Executing Refactor (cvcf_stream.py) on mTOR-Lite..."
/usr/bin/time -v python3 cvcf_stream.py config_massive.txt 2> profiler_ref_mtor.log
exit_4=$?
if [ -f "cvcf_output.txt" ]; then mv cvcf_output.txt cvcf_ref_mtor.txt; fi
extract_telemetry profiler_ref_mtor.log cvcf_ref_mtor.txt $exit_4 4 "cvcf_stream.py" "mTOR-Lite" "N/A"

# --- Output Summary ---
echo -e "\n=========================================================================================="
echo "                           BENCHMARK SUITE COMPLETED                                      "
echo "=========================================================================================="
column -t -s ',' AutoSIM_Audit_Report.csv
echo "=========================================================================================="
echo "Full metrics successfully exported to: AutoSIM_Audit_Report.csv"
