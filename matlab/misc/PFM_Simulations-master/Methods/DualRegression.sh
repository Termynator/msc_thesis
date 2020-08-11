#!/bin/sh

# Runs dual regression on simulated dataset
# Usage: DualRegression.sh <output_dir> <nifti_dir> <group_maps>

# SPDX-License-Identifier: Apache-2.0

output_dir=$1
nifti_dir=$2
group_maps=$3

mkdir "${output_dir}"

# Run all FSL commands locally
unset SGE_ROOT

# Run dual_regression
des_norm=1
design=-1
n_perm=0
dual_regression "${group_maps}" \
    ${des_norm} ${design} ${n_perm} --thr \
    "${output_dir}" \
    $(cat "${nifti_dir}/DualRegression_SpecFile.txt") \
    > "${output_dir}/TerminalOutput.txt"
