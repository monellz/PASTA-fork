#!/bin/bash

declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("freebase_music" "freebase_sampled")
declare -a threads=("2" "4" "8" "16" "32")

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/pasta/coo"

nt=32
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"

for tsr_name in "${test_tsr_names[@]}"
do
	prog_name="dadd_eq"

	# Sequetial code
	dev_id=-2
echo "./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.tns -Y ${tsr_path}/${tsr_name}.tns -d ${dev_id} -c 0 > ${out_path}/${tsr_name}-${prog_name}-c0-seq.txtz"
	# ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-r${R}-seq.txt


	# OpenMP code
	dev_id=-1
	export OMP_NUM_THREADS=${nt}
	# for nt in ${threads[@]}
	# do
		echo "numactl --interleave=0-1 ./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.tns -Y ${tsr_path}/${tsr_name}.tns -d ${dev_id} -c 0 > ${out_path}/${tsr_name}-${prog_name}-c0-t${nt}.txt"
		# numactl --interleave=0-3 ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt
	# done

done