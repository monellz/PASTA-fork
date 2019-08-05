#!/usr/bin/python

from __future__ import print_function

def set_tsrnames():
	#### NEW SET ####
	# For locating data
	s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'flickr', 'freebase_sampled', 'delicious', 'nell1']
	s3tsrs_pl = ['3D_irregular_small', '3D_irregular_medium', '3D_irregular_large', '3D_regular_small', '3D_regular_medium', '3D_regular_large']
	s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'uber-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
	s4tsrs_pl = ['4D_irregular_small', '4D_irregular_medium', '4D_irregular_large', '4D_regular_small', '4D_regular_medium', '4D_regular_large', '4D_i_small', '4D_i_medium', '4D_i_large']

	# For plots
	s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'flickr', 'fb_s', 'deli', 'nell1']
	s3tsrs_pl_names =['irrS', 'irrM', 'irrL', 'regS', 'regM', 'regL']
	s4tsrs_names = ['crime4d', 'nips4d', 'uber4d', 'enron4d', 'flickr4d', 'deli4d']
	s4tsrs_pl_names =['irrS4d', 'irrM4d', 'irrL4d', 'regS4d', 'regM4d', 'regL4d', 'irr2S4d', 'irr2M4d', 'irr2L4d']

	#### OLD SET ####
	# # For locating data
	# s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
	# s3tsrs_pl = ['3D_irregular_small', '3D_irregular_medium', '3D_irregular_large', '3D_regular_small', '3D_regular_medium', '3D_regular_large']
	# s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
	# s4tsrs_pl = ['4D_irregular_small', '4D_irregular_medium', '4D_irregular_large', '4D_regular_small', '4D_regular_medium', '4D_regular_large', '4D_i_small', '4D_i_medium', '4D_i_large']

	# # For plots
	# s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'fb_s', 'deli', 'nell1']
	# s3tsrs_pl_names =['irrS', 'irrM', 'irrL', 'regS', 'regM', 'regL']
	# s4tsrs_names = ['crime4d', 'nips4d', 'enron4d', 'flickr4d', 'deli4d']
	# s4tsrs_pl_names =['irrS4d', 'irrM4d', 'irrL4d', 'regS4d', 'regM4d', 'regL4d', 'irr2S4d', 'irr2M4d', 'irr2L4d']

	return s3tsrs, s3tsrs_pl, s4tsrs, s4tsrs_pl, s3tsrs_names, s3tsrs_pl_names, s4tsrs_names, s4tsrs_pl_names

def get_nnzs(tensors, intput_path):
	nnzs = []
	for tsr in tensors:
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7
		input_str = intput_path + tsr + '_status-b' + str(sb) + '-seq.txt'
		fi = open(input_str, 'r')
		for line in fi:
			# print(line)
			line_array = line.rstrip().split(" ")
			if(len(line_array) >= 3):
				if(line_array[0] == 'NNZ'):
					nnzs.append(int(line_array[2]))

		fi.close()

	return nnzs

def get_nbs(tensors, intput_path):
	nbs = []
	for tsr in tensors:
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7
		input_str = intput_path + tsr + '_status-b' + str(sb) + '-seq.txt'
		fi = open(input_str, 'r')
		for line in fi:
			# print(line)
			line_array = line.rstrip().split(" ")
			if(len(line_array) >= 3):
				if(line_array[0] == 'nb'):
					nbs.append(int(line_array[2]))

		fi.close()

	return nbs

def get_nnzbs(tensors, intput_path):
	nnzbs = []
	for tsr in tensors:
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7
		input_str = intput_path + tsr + '_status-b' + str(sb) + '-seq.txt'
		fi = open(input_str, 'r')
		for line in fi:
			# print(line)
			line_array = line.rstrip().split(" ")
			if(len(line_array) >= 10):
				if(line_array[0] == 'Nnzb:' and line_array[7] == 'Avg'):
					nnzbs.append(int(line_array[9]))

		fi.close()

	return nnzbs

def get_nfibs(tensors, intput_path):
	nfibs = []
	for tsr in tensors:
		nfibs_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7
		input_str = intput_path + tsr + '_status-b' + str(sb) + '-seq.txt'
		fi = open(input_str, 'r')
		for line in fi:
			# print(line)
			line_array = line.rstrip().split(" ")
			if(len(line_array) >= 6):
				if(line_array[4] == 'nfibs:'):
					tmp_arrray = line_array[5].split(",")
					nfibs_modes.append(int(tmp_arrray[0]))
		avg_nfibs = sum(nfibs_modes) / float(len(nfibs_modes))
		nfibs.append(avg_nfibs)

		fi.close()

	return nfibs

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

def comp_actual_gflops(num_flops, times):
	gflops = [ float(num_flops[i]) / times[i] / 1e9 for i in range(len(num_flops)) ]
	return gflops

def comp_actual_gbytes(num_bytes, times):
	gbytes = [ float(num_bytes[i]) / times[i] / 1e9 for i in range(len(num_bytes)) ]
	return gbytes

def predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes, use_cache_bw):
	# compute OI
	oi = [ float(num_flops[i]) / float(num_bytes[i]) for i in range(len(num_flops)) ]
	# predicted gflops
	predicted_gflops = [ comp_gflops_bound(theo_gflops, theo_mem_bw, theo_cache_bw, oi[i], use_cache_bw) for i in range(len(num_flops)) ]

	return oi, predicted_gflops


def comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, times_coo, times_hicoo):
	# actual running time
	gflops_coo = comp_actual_gflops(num_flops, times_coo)
	gflops_hicoo = comp_actual_gflops(num_flops, times_hicoo)
	print_time_array(times_coo, "times_coo")
	print_time_array(times_hicoo, "times_hicoo")
	print("")

	gbytes_coo = comp_actual_gflops(num_bytes_coo, times_coo)
	gbytes_hicoo = comp_actual_gflops(num_bytes_hicoo, times_hicoo)
	print_time_array(gbytes_coo, "gbytes_coo")
	print_time_array(gbytes_hicoo, "gbytes_hicoo")
	print("")

	oi_coo, predicted_gflops_mem_coo = predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, False)
	oi_hicoo, predicted_gflops_mem_hicoo = predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_hicoo, False)

	oi_coo, predicted_gflops_cache_coo = predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, True)
	oi_hicoo, predicted_gflops_cache_hicoo = predict_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_hicoo, True)

	print_time_array(oi_coo, "oi_coo")
	print_time_array(oi_hicoo, "oi_hicoo")
	print("")

	print_gflops_array(gflops_coo, "gflops_coo")
	print_gflops_array(gflops_hicoo, "gflops_hicoo")
	print("")	
	print_time_array(predicted_gflops_mem_coo, "predicted_gflops_mem_coo")
	print_time_array(predicted_gflops_mem_hicoo, "predicted_gflops_mem_hicoo")
	print("")	
	print_time_array(predicted_gflops_cache_coo, "predicted_gflops_cache_coo")
	print_time_array(predicted_gflops_cache_hicoo, "predicted_gflops_cache_hicoo")
	print("\n")

	return gflops_coo, gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo

