#!/usr/bin/python

import sys 
import matplotlib.pyplot as plt
import numpy as np
import common

s3tsrs, s3tsrs_pl, s4tsrs, s4tsrs_pl, s3tsrs_names, s3tsrs_pl_names, s4tsrs_names, s4tsrs_pl_names = common.set_tsrnames()

use_cache_bw_coo_tew = use_cache_bw_hicoo_tew = use_cache_bw_coo_ts = use_cache_bw_hicoo_ts = False
use_cache_bw_coo_ttv = use_cache_bw_hicoo_ttv = use_cache_bw_coo_ttm = use_cache_bw_hicoo_ttm = use_cache_bw_coo_mttkrp = use_cache_bw_hicoo_mttkrp = True

# Global settings for figures
mywidth = 0.35      # the width of the bars

def main(argv):

	if len(argv) < 4:
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

	if plot_tensors == "real":
		tensors = s3tsrs + s4tsrs
	elif plot_tensors == "graph":
		tensors = s3tsrs_pl + s4tsrs_pl
	else:
		print('Wrong plot_tensors!')
		return -1

	print(tensors)

	# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))

	nnzs = common.get_nnzs(tensors, '../new-timing-results/new-timing-results-dgx2/')
	nfibs = common.get_nnzs(tensors, '../new-timing-results/new-timing-results-dgx2/')
	nbs = common.get_nnzs(tensors, '../new-timing-results/new-timing-results-dgx2/')

	####### TEW #########
	op = 'dadd_eq'
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	# rects1, rects2, rects3 = plot_gragh_left(ax1, plot_tensors, "TEW", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(predict_gflops_coo))
	

	# ####### TS #########
	op = 'smul'
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax2, plot_tensors, "TS", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(theo_gflops_array))

	# ####### TTV #########
	op = 'ttv'
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax3, plot_tensors, "TTV", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(theo_gflops_array))
	
	####### TTM #########
	op = 'ttm'
	R = 16
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax4, plot_tensors, "TTM", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(theo_gflops_array))
	# rects1, rects2, rects3 =plot_gragh_modes(ax4, plot_tensors, "", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(theo_gflops_array))

	# ####### MTTKRP #########
	op = 'mttkrp'
	R = 16
	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, nbs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax5, plot_tensors, "MTTKRP", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo), np.asarray(theo_gflops_array))

	# # fig.legend([], ['oral', 'physa'], bbox_to_anchor=(2, 0),loc = 'lower right')
	# # fig.legend(*fig.axes[0,0].get_legend_handles_labels())

	# fig.legend([rects1, rects2, rects3], ["gpu-coo", "gpu-hicoo", "roofline"], loc = 'upper right') # bbox_to_anchor=(0.5, 0)

	# # plt.show()
	# plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')


def plot_gragh_left(ax, plot_tensors, title, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='gpu-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='gpu-hicoo')
	rects3 = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind + mywidth)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects2, rects3


def plot_gragh(ax, plot_tensors, title, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='gpu-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='gpu-hicoo')
	rects3 = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_xticks(ind)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')


def plot_gragh_modes(ax, plot_tensors, title, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		# xnames = s3tsrs_pl_names + s4tsrs_pl_names
		xnames = s3tsrs_pl_names 

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='m1')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='m2')
	rects3 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='m3')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind + mywidth)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects2, rects3


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
		# if tsr in s4tsrs:
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

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo, use_cache_bw_coo_tew, use_cache_bw_hicoo_tew)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo


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
		if tsr in s4tsrs:
		# if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
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

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo, use_cache_bw_coo_tew, use_cache_bw_hicoo_tew)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo


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
		sum_time_modes = 0.0
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
					if(count > 1):
						sum_time += float(line_array[4])
						# print(sum_time)
			fi.close()
			time_num = sum_time / (count - 1)
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		# print(sum_time_modes)
		gpu_times_coo.append(time_num)

		###### HiCOO ######
		# if tsr in s4tsrs:
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time_modes = 0.0
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
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		# print("sum_time_modes:")
		# print(sum_time_modes)
		gpu_times_hicoo.append(time_num)

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

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo, use_cache_bw_coo_tew, use_cache_bw_hicoo_tew)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo


def get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ttm_data")
	gpu_times_coo = []
	gpu_times_hicoo = []
	gpu_mode_times_coo = []
	gpu_mode_times_hicoo = []

	for tsr in tensors:
		# print(tsr)
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
			modes = range(nmodes)
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4
			modes = range(nmodes)

		###### COO ######
		sum_time_modes = 0.0
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
			gpu_mode_times_coo.append(time_num)
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		gpu_mode_times_coo.append(-1)
		# print(sum_time_modes)
		gpu_times_coo.append(time_num)


		###### HiCOO ######
		# if tsr in s4tsrs:
		if tsr in ["chicago-crime-comm-4d", "uber-4d", "nips-4d", 'enron-4d', 'flickr-4d']:
			sb = 4
		else:
			sb = 7

		sum_time_modes = 0.0
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
			gpu_mode_times_hicoo.append(time_num)
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		gpu_mode_times_hicoo.append(-1)
		# print("sum_time_modes:")
		# print(sum_time_modes)
		gpu_times_hicoo.append(time_num)

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	# Calculate GFLOPS and GBytes
	num_flops = [ 2 * i * R for i in nnzs ]
	num_bytes_coo = [ (4 * R * nnzs[i] + 4 * R * nfibs[i] + 8 * nnzs[i] + 8 * nfibs[i]) for i in range(len(nnzs)) ]
	num_bytes_hicoo = num_bytes_coo
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo, use_cache_bw_coo_tew, use_cache_bw_hicoo_tew)

	# print("gpu_mode_times_coo:")
	# print(gpu_mode_times_coo)
	# print("gpu_mode_times_hicoo:")
	# print(gpu_mode_times_hicoo)

	# Calculate GFLOPS
	num_modes_flops = []
	for i in range(len(num_flops)):
		for m in range(nmodes):
			num_modes_flops.append(num_flops[i])
	# print(num_modes_flops)

	# print("gpu_mode_gflops_coo:")
	# print(gpu_mode_gflops_coo)
	# print("gpu_mode_gflops_hicoo:")
	# print(gpu_mode_gflops_hicoo)

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo


def get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, nbs, R, ang_pattern, prefix, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_mttkrp_data")
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
		sum_time_modes = 0.0
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
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		# print(sum_time_modes)
		gpu_times_coo.append(time_num)


		###### HiCOO ######
		# if tsr in s4tsrs:
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time_modes = 0.0
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
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		# print("sum_time_modes:")
		# print(sum_time_modes)
		gpu_times_hicoo.append(time_num)

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	# Calculate GFLOPS and GBytes
	B = pow(2, sb)
	num_flops = [ 3 * i * R for i in nnzs ]
	num_bytes_coo = [ (16 * (R + 1) * nnzs[i]) for i in range(len(nnzs)) ]
	num_bytes_hicoo = [ (16 * nbs[i] * B * R + 7 * nnzs[i] + 20 * nbs[i]) for i in range(len(nnzs)) ]
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, gpu_times_coo, gpu_times_hicoo, use_cache_bw_coo_tew, use_cache_bw_hicoo_tew)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return gpu_gflops_coo, gpu_gflops_hicoo, predicted_gflops_coo, predicted_gflops_hicoo


if __name__ == '__main__':
    sys.exit(main(sys.argv))


