#!/bin/bash

# Script to rename VEP timeseries files by removing task and direction components
# Usage: ./rename_files.sh

# error handling
set -euo pipefail
IFS=$'\n\t'

# set database path
CONTROLS_DB="/controls/data"
PATIENTS_DB="/patients/data"

rename_control_files() {
    local base_dir="$1"
    echo "Processing CONTROL files in: $base_dir"
    
    # timeseries
    find "$base_dir" -type f -name "sub-*_task-rest_dir-*_run-*_vep_timeseries.csv" ! -name "*_labelled*" | while read -r file; do
        dir=$(dirname "$file")
        oldname=$(basename "$file")
        newname=$(echo "$oldname" | sed 's/_task-rest_dir-[A-Z][A-Z]//')
        if [[ "$oldname" != "$newname" ]]; then
            echo "  Timeseries: $oldname -> $newname"
            mv "$file" "$dir/$newname"
        fi
    done

    # atlas
    find "$base_dir" -type f -name "sub-*_task-rest_dir-*_run-*_space-MNI152NLin2009cAsym_desc-vep_dseg.nii"* | while read -r file; do
        dir=$(dirname "$file")
        oldname=$(basename "$file")
        newname=$(echo "$oldname" | sed 's/_task-rest_dir-[A-Z][A-Z]//')
        if [[ "$oldname" != "$newname" ]]; then
            echo "  Atlas: $oldname -> $newname"
            mv "$file" "$dir/$newname"
        fi
    done
}

rename_patient_files() {
    local base_dir="$1"
    echo "Processing PATIENT files in: $base_dir"
    
    # timeseries
    find "$base_dir" -type f -name "sub-*_task-rest_dir-*_run-*_vep_timeseries_labelled.csv" | while read -r file; do
        dir=$(dirname "$file")
        oldname=$(basename "$file")
        newname=$(echo "$oldname" | sed 's/_task-rest_dir-[A-Z][A-Z]//')
        if [[ "$oldname" != "$newname" ]]; then
            echo "  Timeseries: $oldname -> $newname"
            mv "$file" "$dir/$newname"
        fi
    done

    # atlas
    find "$base_dir" -type f -name "sub-*_task-rest_dir-*_run-*_space-MNI152NLin2009cAsym_desc-vep_dseg.nii"* | while read -r file; do
        dir=$(dirname "$file")
        oldname=$(basename "$file")
        newname=$(echo "$oldname" | sed 's/_task-rest_dir-[A-Z][A-Z]//')
        if [[ "$oldname" != "$newname" ]]; then
            echo "  Atlas: $oldname -> $newname"
            mv "$file" "$dir/$newname"
        fi
    done
}

# process controls only
rename_control_files "$CONTROLS_DB"

# process patients only
rename_patient_files "$PATIENTS_DB"

echo "Done!"
