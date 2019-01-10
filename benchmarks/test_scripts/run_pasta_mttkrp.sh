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

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${test_tsr_names[@]}"
	do
		# for mode in ${modes[@]}
		# do

			# Sequetial code
			dev_id=-2
			prog_name="mttkrp"
		echo "./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.tns -Y ${tsr_path}/${tsr_name}.tns -d ${dev_id} -c 0 > ${out_path}/${tsr_name}_$prog_name}-c0-seq.txt"
			./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-r${R}-seq.txt


			# # OpenMP code
			dev_id=-1
			for nt in ${threads[@]}
			do
				# Use reduce
				echo "numactl --interleave=0-3 ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt"
				numactl --interleave=0-3 ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt

				# NOT Use reduce
				# echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} -u 0 > ${out_path}/${tsr_name}-r${R}-t${nt}-noreduce.txt"
				# ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} -u 0 > ${out_path}/${tsr_name}-r${R}-t${nt}-noreduce.txt
			done

		# done
	done
done