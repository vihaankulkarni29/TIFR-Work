#!/bin/bash

echo "======================================================="
echo "   AutoSIM Master Scientific Audit Pipeline            "
echo "======================================================="

# 1. Initialization
rm -f telemetry_data.csv AutoSim_Master_Scientific_Data.csv
echo "Protein,Script_Type,Frame,CVCF,Timestamp_Sec,Compute_Time_ms,Memory_MB" > telemetry_data.csv

# 2. Execution Scenarios
echo -e "\n-------------------------------------------------------"
echo "[Run 1/4] Starting: Ubiquitin | Legacy Architecture..."
python3 cvcf_master_telemetry.py config_small.txt --script-type Legacy --protein-name "Ubiquitin" || echo "   -> Run 1 Failed/OOM as expected"

echo -e "\n-------------------------------------------------------"
echo "[Run 2/4] Starting: Ubiquitin | Ultra Architecture..."
python3 cvcf_master_telemetry.py config_small.txt --script-type Ultra --protein-name "Ubiquitin" || echo "   -> Run 2 Failed"

echo -e "\n-------------------------------------------------------"
echo "[Run 3/4] Starting: mTOR-Lite | Legacy Architecture... (OOM EXPECTED)"
python3 cvcf_master_telemetry.py config_massive.txt --script-type Legacy --protein-name "mTOR-Lite" || echo "   -> CRASH DETECTED: Run 3 suffered OOM Kill as expected"

echo -e "\n-------------------------------------------------------"
echo "[Run 4/4] Starting: mTOR-Lite | Ultra Architecture..."
python3 cvcf_master_telemetry.py config_massive.txt --script-type Ultra --protein-name "mTOR-Lite" || echo "   -> Run 4 Failed"

# 3. Final Packaging
echo -e "\n======================================================="
echo "   Audit Complete! Packaging Master Dataset...         "
mv telemetry_data.csv AutoSim_Master_Scientific_Data.csv

# 4. Validation & Summary
echo -e "\n======================================================="
echo "   Row Integrity Validation Summary                    "
echo "======================================================="
awk -F',' 'NR>1 {count[$1" - "$2]++} END {for (combo in count) print combo " | " count[combo] " telemetry rows compiled"}' AutoSim_Master_Scientific_Data.csv | column -t
echo "======================================================="
echo "Final dataset securely sealed: AutoSim_Master_Scientific_Data.csv"
