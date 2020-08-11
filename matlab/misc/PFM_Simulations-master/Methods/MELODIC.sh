#!/bin/sh

# Runs MELODIC on simulated dataset
# Usage: MELODIC.sh <output_dir> <nifti_dir> <dim> <TR>

# SPDX-License-Identifier: Apache-2.0

output_dir=$1
nifti_dir=$2
dim=$3
TR=$4

mkdir "${output_dir}"

# Run all FSL commands locally
unset SGE_ROOT

# Run MELODIC
melodic -i "${nifti_dir}/MELODIC_SpecFile.txt" \
    -o "${output_dir}" \
    --tr=${TR} -a concat -d ${dim} \
    --migpN=300 --nobet --nomask \
    --maxrestart=5 --verbose \
    > "${output_dir}/TerminalOutput.txt"
