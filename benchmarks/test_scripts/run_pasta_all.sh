#!/bin/bash

echo "./run_pasta_all nmodes nt gpu_dev_id"

nmodes=$1 		# 3, or 4
nt=$2			# 32
gpu_dev_id=$3	# 0, 1, ...

# TEW-eq
./run_pasta_dadd_eq ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_dadd_eq_hicoo ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_ddiv_eq ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_ddiv_eq_hicoo ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_dmul_eq ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_dmul_eq_hicoo ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_dsub_eq ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_dsub_eq_hicoo ${nmodes} ${nt} ${gpu_dev_id}

# TS
./run_pasta_sadd ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_sadd_hicoo ${nmodes} ${nt} ${gpu_dev_id}

./run_pasta_smul ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_smul_hicoo ${nmodes} ${nt} ${gpu_dev_id}

# TTV
./run_pasta_ttv ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_ttv_hicoo ${nmodes} ${nt} ${gpu_dev_id}

# TTM
./run_pasta_ttm ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_ttm_hicoo ${nmodes} ${nt} ${gpu_dev_id}

# MTTKRP
./run_pasta_mttkrp ${nmodes} ${nt} ${gpu_dev_id}
./run_pasta_mttkrp_hicoo ${nmodes} ${nt} ${gpu_dev_id}

