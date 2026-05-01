#!/bin/bash

echo "======================================================="
echo "       AutoSIM DevOps: Deep Audit Pipeline             "
echo "======================================================="

# --- 1. Dynamic Config Generation ---
echo -e "\n[1/3] Generating 'config_audit.txt' dynamically..."
cat << 'EOF' > config_audit.txt
psffile="dummy_mtor.pdb"
pdbfile="dummy_mtor.pdb"
trjfile="dummy_mtor.dcd"
outfile="cvcf_output_audit.txt"
selection="name CA"
align=false
verbose=false
EOF
echo "      -> Configuration file ready."

# --- 2. Execution Engine ---
echo -e "\n[2/3] Executing Telemetry Engine (cvcf_audit_engine.py) on mTOR-Lite..."
python3 cvcf_audit_engine.py config_audit.txt
if [ $? -ne 0 ]; then
    echo "      -> ERROR: Audit Engine crashed or failed to execute!"
    exit 1
fi
echo "      -> Engine successfully generated telemetry."

# --- 3. Verification & The Data Window ---
CSV_FILE="Master_Telemetry_dummy_mtor.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo -e "\n[3/3] ERROR: Expected output file $CSV_FILE was not found!"
    exit 1
fi

echo -e "\n[3/3] The Data Window - Telemetry Verification ($CSV_FILE):"
echo "=========================================================================================="
# Grouping the output of head, a visual separator, and tail together, piping all through column
(
  head -n 6 "$CSV_FILE"
  echo "...,...,...,...,..."
  tail -n 5 "$CSV_FILE"
) | column -s, -t
echo "=========================================================================================="
echo -e "\nDeep Audit Pipeline completed successfully! Full metrics safely written to $CSV_FILE"
