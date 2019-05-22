#!/bin/bash

echo "./run_pasta_all tsr_path out_path nmodes nt gpu_dev_id machine_name"

tsr_path=$1		# "${SCRATCH}/BIGTENSORS"
out_path=$2		# "./timing-results"
nmodes=$3 		# 3, or 4
nt=$4			# 32
gpu_dev_id=-1	# 0, 1, ...
machine_name=$6	# dgx2, wingtip-bigmem2, bluesky

# TS
./benchmarks/test_scripts/run_pasta_smul.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
./benchmarks/test_scripts/run_pasta_smul_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# ./benchmarks/test_scripts/run_pasta_sadd.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
# ./benchmarks/test_scripts/run_pasta_sadd_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# TEW-eq
./benchmarks/test_scripts/run_pasta_dadd_eq.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
./benchmarks/test_scripts/run_pasta_dadd_eq_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# ./benchmarks/test_scripts/run_pasta_ddiv_eq.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
# ./benchmarks/test_scripts/run_pasta_ddiv_eq_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# ./benchmarks/test_scripts/run_pasta_dmul_eq.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
# ./benchmarks/test_scripts/run_pasta_dmul_eq_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# ./benchmarks/test_scripts/run_pasta_dsub_eq.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
# ./benchmarks/test_scripts/run_pasta_dsub_eq_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# TTV
./benchmarks/test_scripts/run_pasta_ttv.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
./benchmarks/test_scripts/run_pasta_ttv_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# TTM
./benchmarks/test_scripts/run_pasta_ttm.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
./benchmarks/test_scripts/run_pasta_ttm_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

# MTTKRP
./benchmarks/test_scripts/run_pasta_mttkrp.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}
./benchmarks/test_scripts/run_pasta_mttkrp_hicoo.sh ${tsr_path} ${out_path} ${nmodes} ${nt} ${gpu_dev_id} ${machine_name}

