#!/usr/bin/python

from __future__ import print_function

def set_tsrnames():
	#### NEW SET ####
	# # For locating data
	# s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'flickr', 'freebase_sampled', 'delicious', 'nell1']
	# s3tsrs_pl = ['3D_irregular_small', '3D_irregular_medium', '3D_irregular_large', '3D_regular_small', '3D_regular_medium', '3D_regular_large']
	# s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'uber-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
	# s4tsrs_pl = ['4D_irregular_small', '4D_irregular_medium', '4D_irregular_large', '4D_regular_small', '4D_regular_medium', '4D_regular_large', '4D_i_small', '4D_i_medium', '4D_i_large']

	# # For plots
	# s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'flickr', 'fb_s', 'deli', 'nell1']
	# s3tsrs_pl_names =['irrS', 'irrM', 'irrL', 'regS', 'regM', 'regL']
	# s4tsrs_names = ['crime4d', 'nips4d', 'uber4d', 'enron4d', 'flickr4d', 'deli4d']
	# s4tsrs_pl_names =['irrS4d', 'irrM4d', 'irrL4d', 'regS4d', 'regM4d', 'regL4d', 'irr2S4d', 'irr2M4d', 'irr2L4d']

	#### OLD SET ####
	# For locating data
	s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
	s3tsrs_pl = ['3D_irregular_small', '3D_irregular_medium', '3D_irregular_large', '3D_regular_small', '3D_regular_medium', '3D_regular_large']
	s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
	s4tsrs_pl = ['4D_irregular_small', '4D_irregular_medium', '4D_irregular_large', '4D_regular_small', '4D_regular_medium', '4D_regular_large', '4D_i_small', '4D_i_medium', '4D_i_large']

	# For plots
	s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'fb_s', 'deli', 'nell1']
	s3tsrs_pl_names =['irrS', 'irrM', 'irrL', 'regS', 'regM', 'regL']
	s4tsrs_names = ['crime4d', 'nips4d', 'enron4d', 'flickr4d', 'deli4d']
	s4tsrs_pl_names =['irrS4d', 'irrM4d', 'irrL4d', 'regS4d', 'regM4d', 'regL4d', 'irr2S4d', 'irr2M4d', 'irr2L4d']

	return s3tsrs, s3tsrs_pl, s4tsrs, s4tsrs_pl, s3tsrs_names, s3tsrs_pl_names, s4tsrs_names, s4tsrs_pl_names

def get_nnzs(tensors, intput_path):
	nnzs = []
	for tsr in tensors:
		# Get NNZ
		count = 0
		input_str = intput_path + tsr + '_dadd_eq-seq.txt'
		fi = open(input_str, 'r')
		for line in fi:
			# print(line)
			line_array = line.rstrip().split(" ")
			if(len(line_array) > 1):
				tmp_arrray = line_array[1].split("=")
				if(tmp_arrray[0] == 'NNZ' and count == 0):
					count += 1
					nnzs.append(int(tmp_arrray[1]))

		fi.close()

	return nnzs

def comp_gflops_bound(theo_gflops, theo_mem_bw, theo_cache_bw, oi_alg, use_cache_bw):
	oi_mem = theo_gflops / theo_mem_bw
	oi_cache = theo_gflops / theo_cache_bw
	# print("oi_mem: " + str(oi_mem))
	# print("oi_cache: " + str(oi_cache))
	assert (oi_mem >= oi_cache)

	if oi_alg >= oi_mem:
		gflops_bound = theo_gflops
	elif oi_alg > oi_cache and oi_alg < oi_mem:
		gflops_bound = oi_alg * theo_mem_bw
	elif oi_alg <= oi_cache:
		if use_cache_bw == True:
			gflops_bound = oi_alg * theo_cache_bw
		else:
			gflops_bound = oi_alg * theo_mem_bw

	return gflops_bound

def print_gflops_array(array, array_name):
	print(array_name+":")
	for i in array:
		print("%.2f, " % i, end='')
	print("")

def print_time_array(array, array_name):
	print(array_name+":")
	for i in array:
		print("%.4f, " % i, end='')
	print("")

def comp_actual_gflops(num_flops, gpu_times):
	gpu_gflops = [ float(num_flops[i]) / gpu_times[i] / 1e9 for i in range(len(num_flops)) ]
	return gpu_gflops

def predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes, use_cache_bw):
	# compute OI
	oi = [ float(num_flops[i]) / float(num_bytes[i]) for i in range(len(num_flops)) ]
	# predicted gflops
	predicted_gflops = [ comp_gflops_bound(theo_gflops, theo_mem_bw, theo_cache_bw, oi[i], use_cache_bw) for i in range(len(num_flops)) ]

	return oi, predicted_gflops


def comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo, use_cache_bw_coo, use_cache_bw_hicoo):
	# actual running time
	gpu_gflops_coo = comp_actual_gflops(num_flops, gpu_times_coo)
	gpu_gflops_hicoo = comp_actual_gflops(num_flops, gpu_times_hicoo)
	print_time_array(gpu_times_coo, "gpu_times_coo")
	print_time_array(gpu_times_hicoo, "gpu_times_hicoo")
	print("")

	oi_coo, predicted_gflops_coo = predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, use_cache_bw_coo)
	oi_hicoo, predicted_gflops_hicoo = predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_hicoo, use_cache_bw_hicoo)

	print_time_array(oi_coo, "oi_coo")
	print_time_array(oi_hicoo, "oi_hicoo")
	print("")

	print_gflops_array(gpu_gflops_coo, "gpu_gflops_coo")
	print_gflops_array(gpu_gflops_hicoo, "gpu_gflops_hicoo")
	print("")	
	print_time_array(predicted_gflops_coo, "predicted_gflops_coo")
	print_time_array(predicted_gflops_hicoo, "predicted_gflops_hicoo")
	print("\n")

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo

