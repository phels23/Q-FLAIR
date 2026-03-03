#!/bin/bash

# Check if arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_name> <max_nGate>"
    echo "Example: $0 bars-and-stripes_logLoss 10"
    exit 1
fi

FOLDER=$1
MAX_NGATE=$2
ORIGINAL_PARAMS="$FOLDER/params"

# Check if the params file actually exists
if [ ! -f "$ORIGINAL_PARAMS" ]; then
    echo "Error: File '$ORIGINAL_PARAMS' not found!"
    exit 1
fi

# Loop from 1 to MAX_NGATE
for (( i=1; i<=MAX_NGATE; i++ ))
do
    NEW_PARAMS="$FOLDER/params_$i"

    echo "----------------------------------------"
    echo "Preparing nGate = $i"

    # 1. Duplicate the file
    cp "$ORIGINAL_PARAMS" "$NEW_PARAMS"

    # 2. Append the new lines
    # We add a newline first to ensure we aren't appending to the end of an existing line
    echo "" >> "$NEW_PARAMS"
    echo "nGate $i" >> "$NEW_PARAMS"
    echo "accOut      $FOLDER/ACC_classical_surrogate_$i.txt" >> "$NEW_PARAMS"

    # 3. Run the Python script with the NEW params file
    echo "Running benchmark with $NEW_PARAMS..."
    python QFLAIR_QNN_benchmark.py -p "$NEW_PARAMS"
done

echo "Done!"
