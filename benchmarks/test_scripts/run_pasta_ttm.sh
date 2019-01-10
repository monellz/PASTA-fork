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
prog_name="ttm"

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${test_tsr_names[@]}"
	do
		for mode in ${modes[@]}
		do

			# Sequetial code
			dev_id=-2
			echo "./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}_${prog_name}}-m${mode}-r${R}-seq.txt"
			./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}_${prog_name}}-m${mode}-r${R}-seq.txt


			# # OpenMP code
			dev_id=-1
			export OMP_NUM_THREADS=${nt}
			# for nt in ${threads[@]}
			# do
				echo "numactl --interleave=0-1 ./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}_${prog_name}}-m${mode}-r${R}-t${nt}.txt"
				numactl --interleave=0-1 ./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}_${prog_name}}-m${mode}-r${R}-t${nt}.txt
			# done

		done
	done
done