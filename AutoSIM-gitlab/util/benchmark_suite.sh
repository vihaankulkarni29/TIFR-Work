#!/bin/bash

echo "======================================================="
echo "  AutoSIM DevOps Pipeline: Deep Telemetry Audit Suite  "
echo "======================================================="

# 1. Dynamically Generate Configs (align=false fixes the PBC bug)
cat << 'EOF' > config_small.txt
psffile="ubiquitin.pdb"
pdbfile="ubiquitin.pdb"
trjfile="ubiquitin.dcd"
outfile="cvcf_output_small.txt"
selection="name CA"
align=false
verbose=false
EOF

cat << 'EOF' > config_massive.txt
psffile="dummy_mtor.pdb"
pdbfile="dummy_mtor.pdb"
trjfile="dummy_mtor.dcd"
outfile="cvcf_output_massive.txt"
selection="name CA"
align=false
verbose=false
EOF

# 2. Initialize CSV Log
CSV_FILE="AutoSIM_Audit_Report.csv"
echo "Stage,Script,Target_Molecule,Peak_RAM_MB,Speed_Sec,Output_Size,Exit_Status,Math_Parity" > $CSV_FILE

# 3. Execution Engine
run_test() {
    STAGE=$1
    SCRIPT=$2
    TARGET=$3
    CONFIG=$4

    echo -e "\n[Stage $STAGE/4] Executing $SCRIPT on $TARGET..."
    
    # Get the intended output filename from config
    OUT_FILE=$(grep outfile $CONFIG | cut -d'"' -f2)
    rm -f $OUT_FILE profiler.log diff.log
    
    # Run the profiler
    /usr/bin/time -v python3 $SCRIPT $CONFIG 2> profiler.log
    EXIT_CODE=$?
    
    # Parse RAM
    RAM_KB=$(grep "Maximum resident set size" profiler.log | awk '{print $6}')
    RAM_MB=$(awk "BEGIN {printf \"%.2f\", $RAM_KB/1024}")
    
    # Parse CPU Time (User + System)
    USER_TIME=$(grep "User time (seconds)" profiler.log | awk '{print $4}')
    SYS_TIME=$(grep "System time (seconds)" profiler.log | awk '{print $4}')
    SPEED=$(awk "BEGIN {printf \"%.2f\", $USER_TIME + $SYS_TIME}")
    
    # Check Status
    if [ $EXIT_CODE -ne 0 ] || [ ! -f "$OUT_FILE" ]; then
        STATUS="CRASH_(OOM)"
        SIZE="N/A"
    else
        STATUS="SUCCESS"
        SIZE=$(ls -lh $OUT_FILE | awk '{print $5}')
    fi
    
    # Save variables dynamically for the parity check
    eval "OUT_${STAGE}=${OUT_FILE}"
    eval "STAT_${STAGE}=${STATUS}"
    
    # Calculate Parity
    PARITY="N/A"
    if [ "$STAGE" == "2" ]; then
        if [ "$STAT_1" == "SUCCESS" ] && [ "$STATUS" == "SUCCESS" ]; then
            diff $OUT_1 $OUT_FILE > diff.log
            if [ -s diff.log ]; then PARITY="FAIL"; else PARITY="100%_MATCH"; fi
        fi
    elif [ "$STAGE" == "4" ]; then
         if [ "$STAT_3" == "SUCCESS" ] && [ "$STATUS" == "SUCCESS" ]; then
            diff $OUT_3 $OUT_FILE > diff.log
            if [ -s diff.log ]; then PARITY="FAIL"; else PARITY="100%_MATCH"; fi
        elif [ "$STAT_3" == "CRASH_(OOM)" ] && [ "$STATUS" == "SUCCESS" ]; then
            PARITY="SURVIVED_OOM"
        fi
    else
        PARITY="Baseline"
    fi

    # Append to CSV
    echo "$STAGE,$SCRIPT,$TARGET,$RAM_MB,$SPEED,$SIZE,$STATUS,$PARITY" >> $CSV_FILE
}

# 4. Run the Gauntlet
run_test 1 "cvcf.py" "Ubiquitin" "config_small.txt"
run_test 2 "cvcf_stream.py" "Ubiquitin" "config_small.txt"
run_test 3 "cvcf.py" "mTOR-Lite" "config_massive.txt"
run_test 4 "cvcf_stream.py" "mTOR-Lite" "config_massive.txt"

# 5. Output Final Results
echo -e "\n=========================================================================================="
echo "                            BENCHMARK SUITE COMPLETED                                     "
echo "=========================================================================================="
column -s, -t < $CSV_FILE
echo "=========================================================================================="
echo "Metrics exported safely to: $CSV_FILE"
