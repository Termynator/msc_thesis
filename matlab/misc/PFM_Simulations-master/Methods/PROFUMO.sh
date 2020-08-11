#!/bin/sh

# Runs PROFUMO on simulated dataset
# Usage: PROFUMO.sh <output_dir> <nifti_dir> <dim> <TR>

# SPDX-License-Identifier: Apache-2.0

output_dir=$1
nifti_dir=$2
dim=$3
TR=$4

mkdir "${output_dir}"

# Run PROFUMO (need to make sure we're on jalapeno18!)
#nice -n 20 PROFUMO \
PROFUMO \
    "${nifti_dir}/PROFUMO_SpecFile.json" \
    ${dim} \
    "${output_dir}" \
    --useHRF ${TR} --hrfFile ~samh/PROFUMO/HRFs/Default.phrf \
    --covModel Run --lowRankData 150 -d 0.5 --globalInit --nThreads 10 \
    > "${output_dir}/TerminalOutput.txt"
