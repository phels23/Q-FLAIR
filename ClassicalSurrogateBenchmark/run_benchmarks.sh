#!/bin/bash

# 1. Check if the user provided the max_nGate argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <max_nGate>"
    echo "Example: $0 10"
    exit 1
fi

MAX_NGATE=$1
WORKER_SCRIPT="./run_benchmark.sh"

# 2. Check if the worker script exists and is executable
if [ ! -x "$WORKER_SCRIPT" ]; then
    echo "Error: $WORKER_SCRIPT not found or not executable."
    echo "Run: chmod +x run_benchmark.sh"
    exit 1
fi

echo "Searching for folders containing a 'params' file..."
echo "--------------------------------------------------"

FOUND_ANY=false

# 3. Loop through all directories in the current location
for dir in */ ; do
    # Remove the trailing slash (e.g., "folder/" -> "folder")
    FOLDER_NAME=${dir%/}

    # Check if 'params' exists inside this folder
    if [ -f "$FOLDER_NAME/params" ]; then
        FOUND_ANY=true
        echo ">>> Found valid folder: '$FOLDER_NAME'. Starting benchmark..."

        # Call the worker script
        $WORKER_SCRIPT "$FOLDER_NAME" "$MAX_NGATE"

        echo "--------------------------------------------------"
    fi
done

if [ "$FOUND_ANY" = false ]; then
    echo "No folders containing a 'params' file were found in this directory."
else
    echo "All benchmarks finished."
fi