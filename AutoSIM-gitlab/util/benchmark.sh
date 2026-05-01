#!/bin/bash

echo "==================================================="
echo "  AutoSIM Memory Optimization Benchmark Suite      "
echo "==================================================="

# 1. Clean the workspace of old test data
rm -f cvcf_mitradip.txt cvcf_vihaan.txt diff_log.txt profiler_mitradip.log profiler_vihaan.log

# 2. Run the Control (Dr. Mitradip's original code)
echo "[1/3] Executing Control: cvcf.py..."
# We pipe the time output (stderr) to a dedicated log file
/usr/bin/time -v python3 cvcf.py config.txt 2> profiler_mitradip.log
# Rename the standard output so it doesn't get overwritten
if [ -f "cvcf_output.txt" ]; then
    mv cvcf_output.txt cvcf_mitradip.txt
else
    echo "  -> ERROR: Control run failed to produce output."
fi

# 3. Run the Refactor (Your streaming code)
echo -e "\n[2/3] Executing Refactor: cvcf_stream.py..."
/usr/bin/time -v python3 cvcf_stream.py config.txt 2> profiler_vihaan.log
if [ -f "cvcf_output.txt" ]; then
    mv cvcf_output.txt cvcf_vihaan.txt
else
    echo "  -> ERROR: Refactor run failed to produce output."
fi

# 4. The Mathematical Parity Test
echo -e "\n[3/3] Running Diff Engine for Mathematical Parity..."
if [ -f "cvcf_mitradip.txt" ] && [ -f "cvcf_vihaan.txt" ]; then
    diff cvcf_mitradip.txt cvcf_vihaan.txt > diff_log.txt
    
    echo "==================================================="
    echo "  Benchmark Complete                               "
    echo "==================================================="
    
    # Check if the diff file has any content
    if [ -s diff_log.txt ]; then
        echo "WARNING: Math discrepancies found. Check diff_log.txt"
    else
        echo "SUCCESS: 100% Mathematical Parity Achieved!"
    fi
else
    echo "Diff skipped due to missing output files (check crash logs)."
fi

# Extract and display just the RAM usage for a quick visual check
echo -e "\n--- RAM Footprint Summary ---"
grep "Maximum resident set size" profiler_mitradip.log | sed 's/^/Control : /'
grep "Maximum resident set size" profiler_vihaan.log | sed 's/^/Refactor: /'
