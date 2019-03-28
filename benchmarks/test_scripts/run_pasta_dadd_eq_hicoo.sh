#!/bin/bash

declare -a s3tsrs=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "freebase_sampled" "delicious" "nell1")
# declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("vast-2015-mc1")
# declare -a threads=("2" "4" "8" "16" "32")

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/SpTenBench/timing-results/pasta/coo"

nmodes=$1 		# 3, or 4
nt=$2			# 32
gpu_dev_id=$3	# 0, 1, ...

prog_name="dadd_eq_hicoo_gpu"
sb=7
if [[ ${nmodes} = "3" ]]; then
	run_tsrs=("${s3tsrs[@]}") 
elif [[ ${nmodes} = "4" ]]; then
	run_tsrs=("${s4tsrs[@]}") 
fi

for tsr_name in "${run_tsrs[@]}"
do
	if [ ${tsr_name} = "chicago-crime-comm-4d" ] || [ ${tsr_name} = "uber-4d" ]; then
		sb=4
	fi

	# Sequetial code
	dev_id=-2
	myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}_nnz.tns -Y ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -b ${sb} > ${out_path}/${tsr_name}_${prog_name}-b${sb}-seq.txt"
	echo ${myprogram}
	# ${myprogram}

	# OpenMP code
	dev_id=-1
	export OMP_NUM_THREADS=${nt}
	myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}_nnz.tns -Y ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -b ${sb} > ${out_path}/${tsr_name}_${prog_name}-b${sb}-t${nt}.txt"
	echo ${myprogram}
	# ${myprogram}

	# CUDA code
	dev_id=${gpu_dev_id}
	myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}_nnz.tns -Y ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -b ${sb} > ${out_path}/${tsr_name}_${prog_name}-b${sb}-gpu.txt"
	echo ${myprogram}
	# ${myprogram}

	echo 

done