#!/bin/bash

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --object) object="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if the --object parameter was provided
if [ -z "$object" ]; then
    echo "Error: --object parameter is required"
    exit 1
fi

# Use the 'object' variable in your script
python stage1.py --object "$object" >> logs/"$object".log

python stage2.py --object "$object" >> logs/"$object".log

python stage3.py --object "$object" >> logs/"$object".log

python stage4.py --object "$object" >> logs/"$object".log
