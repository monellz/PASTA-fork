#!/bin/bash

if [[ $# < 5 ]]; then
	echo "./run_pasta_all_hicoo tsr_path out_path nt gpu_dev_id machine_name"
	exit
fi

tsr_path=$1		# "${SCRATCH}/BIGTENSORS"
out_path=$2		# "./timing-results"
nt=$3			# 32
gpu_dev_id=$4	# 0, 1, ...
machine_name=$5	# dgx2, wingtip-bigmem2, bluesky

echo "./run_pasta_all_coo ${tsr_path} ${out_path} ${nt} ${gpu_dev_id} ${machine_name}"
echo

script_path="./benchmarks/test_scripts"

declare -a dev_ids=("-2" "-1" ${gpu_dev_id})	# Need to modify for platform
declare -a modes=("3" "4")

for id in "${dev_ids[@]}"
do
	for nmodes in "${modes[@]}"
	do
		apped="${tsr_path} ${out_path} ${nmodes} ${nt} ${id} ${machine_name}"

		# TS
		${script_path}/run_pasta_smul.sh ${apped}

		# TEW-eq
		${script_path}/run_pasta_dadd_eq.sh ${apped}

		# TTV
		${script_path}/run_pasta_ttv.sh ${apped}

		# TTM
		${script_path}/run_pasta_ttm.sh ${apped}

		# MTTKRP
		${script_path}/run_pasta_mttkrp.sh ${apped}

	done
done