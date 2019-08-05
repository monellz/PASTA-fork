#!/usr/bin/python

import sys 
import matplotlib.pyplot as plt
import numpy as np
import common
import plots

s3tsrs, s3tsrs_pl, s4tsrs, s4tsrs_pl, s3tsrs_names, s3tsrs_pl_names, s4tsrs_names, s4tsrs_pl_names = common.set_tsrnames()

def main(argv):

	if len(argv) < 5:
		print("Usage: %s intput_path plot_tensors(real/graph) ang_pattern(0/1-dgx1/2-dgx2) machine_name(dgx1,dgx2)" % argv[0])
		exit(-1)

	# input parameters
	intput_path = sys.argv[1]
	plot_tensors = sys.argv[2]
	ang_pattern = sys.argv[3]
	machine_name = sys.argv[4]
	print('intput_path: %s' % intput_path)
	print('plot_tensors: %s' % plot_tensors)
	print('ang_pattern: %s' % ang_pattern)
	print('machine_name: %s' % machine_name)

	if ang_pattern == '1':
		prefix = "dgx-1_"
	elif ang_pattern == '2':
		prefix = "dgx-2_"
	else:
		prefix = ""

	# theoretical machine numbers
	if machine_name == "dgx1":
		# p100
		theo_gflops = 10600.0
		theo_mem_bw = 517.0
		theo_cache_bw = 1920.0
	elif machine_name == "dgx2":
		# v100 
		theo_gflops = 14900.0
		theo_mem_bw = 791.0
		theo_cache_bw = 3332.0
	else:
		print('Wrong machine_name!')
		return -1
	print('theo_gflops: %.2f, theo_mem_bw: %.2f, theo_cache_bw: %.2f' % (theo_gflops, theo_mem_bw, theo_cache_bw))

	if plot_tensors == "real":
		tensors = s3tsrs + s4tsrs
	elif plot_tensors == "graph":
		tensors = s3tsrs_pl + s4tsrs_pl
	else:
		print('Wrong plot_tensors!')
		return -1

	print("tensors:")
	print(tensors)

	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))

	nnzs = common.get_nnzs(tensors, '../new-timing-results/new-timing-results-wingtip-bigmem2/')
	nfibs = common.get_nfibs(tensors, '../new-timing-results/new-timing-results-wingtip-bigmem2/')
	nbs = common.get_nbs(tensors, '../new-timing-results/new-timing-results-wingtip-bigmem2/')
	nnzbs = common.get_nnzbs(tensors, '../new-timing-results/new-timing-results-wingtip-bigmem2/')
	assert(len(nnzs) == len(nfibs))
	assert(len(nnzs) == len(nbs))
	assert(len(nnzs) == len(nnzbs))

	print("\nnnzs:")
	print(nnzs)
	print("nfibs:")
	print(nfibs)
	print("nbs:")
	print(nbs)
	print("nnzbs:")
	print(nnzbs)
	print("")

	####### TEW #########
	op = 'dadd_eq'
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	rects1, rects2, rects3 = plots.plot_gragh_left(ax1, plot_tensors, "TEW", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predicted_gflops_mem_coo))
	

	# ####### TS #########
	op = 'smul'
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	plots.plot_gragh(ax2, plot_tensors, "TS", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predicted_gflops_mem_coo))

	# ####### TTV #########
	op = 'ttv'
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	plots.plot_gragh(ax3, plot_tensors, "TTV", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predicted_gflops_mem_coo))
	
	####### TTM #########
	op = 'ttm'
	R = 16
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	plots.plot_gragh(ax4, plot_tensors, "TTM", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predicted_gflops_mem_coo))
	# rects1, rects2, rects3 =plot_gragh_modes(ax4, plot_tensors, "", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predicted_gflops_mem_coo))

	# ####### MTTKRP #########
	op = 'mttkrp'
	R = 16
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, nbs, nnzbs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	plots.plot_gragh(ax5, plot_tensors, "MTTKRP", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predicted_gflops_mem_coo))

	# # fig.legend([], ['oral', 'physa'], bbox_to_anchor=(2, 0),loc = 'lower right')
	# # fig.legend(*fig.axes[0,0].get_legend_handles_labels())

	fig.legend([rects1, rects2, rects3], ["gpu-coo", "gpu-hicoo", "roofline"], loc = 'upper right') # bbox_to_anchor=(0.5, 0)

	plt.show()
	# plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')


def get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("\n### get_tew_data ###")
	gpu_times_coo = []
	gpu_times_hicoo = []

	for tsr in tensors:
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4

		###### COO ######
		sum_time = 0.0
		count = 0
		if ang_pattern == '1' or ang_pattern == '2':
			input_str = intput_path + prefix + tsr + '_' + op + '_Mode' + str(nmodes) + '_gpu.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '_gpu-gpu.txt'
		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			# print line_array
			if(len(line_array) < 4):
				continue;
			elif(line_array[2] == 'DotAdd]:'):
				count += 1
				if(count > 1):
					sum_time += float(line_array[3])
					# print(sum_time)
		fi.close()
		time_num = sum_time / (count - 1)
		gpu_times_coo.append(time_num)


		###### HiCOO ######
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time = 0.0
		count = 0
		if ang_pattern == '1' or ang_pattern == '2':
			input_str = intput_path + prefix + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_gpu.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '_hicoo_gpu-b' + str(sb) + '-gpu.txt'
		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			# print line_array
			if(len(line_array) < 4):
				continue;
			elif(line_array[2] == 'DotAdd]:'):
				count += 1
				if(count > 1):
					sum_time += float(line_array[3])
					# print(sum_time)
		fi.close()
		time_num = sum_time / (count - 1)
		gpu_times_hicoo.append(time_num)


	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	# Calculate work (#Flops) and memory access (#Bytes)
	num_flops = nnzs
	num_bytes_coo = [ 12 * nnzs[i] for i in range(len(nnzs)) ]
	num_bytes_hicoo = num_bytes_coo
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ts_data")
	gpu_times_coo = []
	gpu_times_hicoo = []

	for tsr in tensors:
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4

		###### COO ######
		sum_time = 0.0
		count = 0
		if ang_pattern == '1' or ang_pattern == '2':
			input_str = intput_path + prefix + tsr + '_' + op + '_Mode' + str(nmodes) + '_gpu.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '_gpu-gpu.txt'
		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			# print line_array
			if(len(line_array) < 4):
				continue;
			elif(line_array[2] == 'MulScalar]:'):
				count += 1
				if(count > 1):
					sum_time += float(line_array[3])
					# print(sum_time)
		fi.close()
		time_num = sum_time / (count - 1)
		gpu_times_coo.append(time_num)

		###### HiCOO ######
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time = 0.0
		count = 0
		if ang_pattern == '1' or ang_pattern == '2':
			input_str = intput_path + prefix + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_gpu.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '_hicoo_gpu-b' + str(sb) + '-gpu.txt'
		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			# print line_array
			if(len(line_array) < 4):
				continue;
			elif(line_array[2] == 'MulScalar]:'):
				count += 1
				if(count > 1):
					sum_time += float(line_array[3])
					# print(sum_time)
		fi.close()
		time_num = sum_time / (count - 1)
		gpu_times_hicoo.append(time_num)


	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	# Calculate GFLOPS and GBytes
	num_flops = nnzs
	num_bytes_coo = [ 8 * nnzs[i] for i in range(len(nnzs)) ]
	num_bytes_hicoo = num_bytes_coo
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ttv_data")
	gpu_times_coo = []
	gpu_times_hicoo = []

	for tsr in tensors:
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
			modes = range(nmodes)
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4
			modes = range(nmodes)

		###### COO ######
		time_modes = []
		for m in modes:
			sum_time = 0.0
			count = 0
			if ang_pattern == '1' or ang_pattern == '2':
				input_str = intput_path + prefix + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_gpu.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_gpu-m' + str(m) + '-gpu.txt'
			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				# print line_array
				if(len(line_array) < 4):
					continue;
				elif(line_array[3] == 'Vec]:'):
					count += 1
					if(count > 1):	# Skip the first warm-up execution
						sum_time += float(line_array[4])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			time_modes.append(time_num)
		# print("time_modes coo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		gpu_times_coo.append(avg_time_modes)	# could use min_time_modes

		###### HiCOO ######
		time_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		for m in modes:
			sum_time = 0.0
			count = 0
			if ang_pattern == '1' or ang_pattern == '2':
				input_str = intput_path + prefix + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_gpu.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo_gpu-m' + str(m) + '-b' + str(sb) + '-gpu.txt'
			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				# print line_array
				if(len(line_array) < 4):
					continue;
				elif(line_array[3] == 'Vec]:'):
					count += 1
					if(count > 1):
						sum_time += float(line_array[4])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			time_modes.append(time_num)
		# print("time_modes hicoo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		gpu_times_hicoo.append(avg_time_modes)	# could use min_time_modes

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	# Calculate GFLOPS and GBytes
	num_flops = [ 2 * i for i in nnzs ]
	num_bytes_coo = [ (12 * nnzs[i] + 16 * nfibs[i]) for i in range(len(nnzs)) ]
	num_bytes_hicoo = num_bytes_coo
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	# Predict using memory BW
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ttm_data")
	gpu_times_coo = []
	gpu_times_hicoo = []
	# gpu_mode_times_coo = []
	# gpu_mode_times_hicoo = []

	for tsr in tensors:
		# print(tsr)
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
			modes = range(nmodes)
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4
			modes = range(nmodes)

		###### COO ######
		time_modes = []
		for m in modes:
			sum_time = 0.0
			count = 0
			if ang_pattern == '1' or ang_pattern == '2':
				input_str = intput_path + prefix + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_gpu.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_gpu-m' + str(m) + '-r' + str(R) + '-gpu.txt'
			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				# print line_array
				if(len(line_array) < 4):
					continue;
				elif(line_array[3] == 'Mtx]:'):
					count += 1
					if(count > 1):
						sum_time += float(line_array[4])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			time_modes.append(time_num)
			# gpu_mode_times_coo.append(time_num)
		# print("time_modes coo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		gpu_times_coo.append(avg_time_modes)	# could use min_time_modes
		# gpu_mode_times_coo.append(-1)


		###### HiCOO ######
		time_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		for m in modes:
			sum_time = 0.0
			count = 0
			if ang_pattern == '1' or ang_pattern == '2':
				input_str = intput_path + prefix + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_gpu.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo_gpu-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-gpu.txt'
			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				# print line_array
				if(len(line_array) < 4):
					continue;
				elif(line_array[3] == 'Mtx]:'):
					count += 1
					if(count > 1):
						sum_time += float(line_array[4])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			# gpu_mode_times_hicoo.append(time_num)
			time_modes.append(time_num)
		# print("time_modes hicoo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		gpu_times_hicoo.append(avg_time_modes)	# could use min_time_modes
		# gpu_mode_times_hicoo.append(-1)

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	# Calculate GFLOPS and GBytes
	num_flops = [ 2 * i * R for i in nnzs ]
	num_bytes_coo = [ (4 * R * (nnzs[i] + nfibs[i]) + 8 * (nnzs[i] + nfibs[i])) for i in range(len(nnzs)) ]
	num_bytes_hicoo = num_bytes_coo
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo)

	# print("gpu_mode_times_coo:")
	# print(gpu_mode_times_coo)
	# print("gpu_mode_times_hicoo:")
	# print(gpu_mode_times_hicoo)

	# Calculate GFLOPS
	# num_modes_flops = []
	# for i in range(len(num_flops)):
	# 	for m in range(nmodes):
	# 		num_modes_flops.append(num_flops[i])
	# print(num_modes_flops)

	# print("gpu_mode_gflops_coo:")
	# print(gpu_mode_gflops_coo)
	# print("gpu_mode_gflops_hicoo:")
	# print(gpu_mode_gflops_hicoo)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, nbs, nnzbs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_mttkrp_data")
	gpu_times_coo = []
	gpu_times_hicoo = []
	num_flops = []
	num_bytes_coo = []
	num_bytes_hicoo = []
	tsr_count = 0

	for tsr in tensors:
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
			modes = range(nmodes)
			num_bytes_coo.append(16 * nnzs[tsr_count] * (R + 1))
			num_bytes_hicoo.append(16 * min(nbs[tsr_count] * nnzbs[tsr_count], nnzs[tsr_count]) * R + 7 * nnzs[tsr_count] + 20 * nbs[tsr_count])
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4
			modes = range(nmodes)
			num_bytes_coo.append(20 * nnzs[tsr_count] * (R + 1))
			num_bytes_hicoo.append(20 * min(nbs[tsr_count] * nnzbs[tsr_count], nnzs[tsr_count]) * R + 8 * nnzs[tsr_count] + 24 * nbs[tsr_count])
		num_flops.append(nmodes * R * nnzs[tsr_count])
		tsr_count += 1

		###### COO ######
		time_modes = []
		for m in modes:
			sum_time = 0.0
			count = 0
			if ang_pattern == '1' or ang_pattern == '2':
				input_str = intput_path + prefix + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_gpu.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_gpu-m' + str(m) + '-r' + str(R) + '-gpu.txt'
			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				# print line_array
				if(len(line_array) < 4):
					continue;
				elif(line_array[2] == 'MTTKRP]:'):
					count += 1
					if(count > 1):
						sum_time += float(line_array[3])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			time_modes.append(time_num)
		# print("time_modes coo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		gpu_times_coo.append(avg_time_modes)	# could use min_time_modes


		###### HiCOO ######
		time_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		for m in modes:
			sum_time = 0.0
			count = 0
			# input_str = '../timing-results-marianas/' + tsr + '_' + op + '_hicoo_gpu-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-gpu.txt'
			if ang_pattern == '1' or ang_pattern == '2':
				input_str = intput_path + prefix + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_gpu.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo_gpu-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-gpu.txt'
			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				# print line_array
				if(len(line_array) < 4):
					continue;
				elif(line_array[2] == 'MTTKRP]:'):
					count += 1
					if(count > 1):
						sum_time += float(line_array[3])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			time_modes.append(time_num)
		# print("time_modes hicoo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		gpu_times_hicoo.append(avg_time_modes)	# could use min_time_modes

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))
	assert(tsr_count == len(nnzs))

	# Calculate GFLOPS and GBytes
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


if __name__ == '__main__':
    sys.exit(main(sys.argv))


