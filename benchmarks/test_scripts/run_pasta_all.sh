#!/bin/bash

echo "./run_pasta_all tsr_path out_path nmodes nt gpu_dev_id"

tsr_path=$1		# "${SCRATCH}/BIGTENSORS"
out_path=$2		# "./timing-results"
nmodes=$3 		# 3, or 4
nt=$4			# 32
gpu_dev_id=$5	# 0, 1, ...

# MTTKRP
./run_pasta_mttkrp ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_mttkrp_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

# TS
./run_pasta_sadd ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_sadd_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_smul ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_smul_hicoo ${tsr_path} ${out_path} {nmodes} ${nt} ${gpu_dev_id}

# TTV
./run_pasta_ttv ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_ttv_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

# TTM
./run_pasta_ttm ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_ttm_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

# TEW-eq
./run_pasta_dadd_eq ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_dadd_eq_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_ddiv_eq ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_ddiv_eq_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_dmul_eq ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_dmul_eq_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_dsub_eq ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_dsub_eq_hicoo ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id}
