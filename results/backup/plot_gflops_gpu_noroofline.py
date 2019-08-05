#!/usr/bin/python

import sys 
import matplotlib.pyplot as plt
import numpy as np

# For locating data
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
# s3tsrs_pl = ['3D_irregular_large', '3D_irregular_medium', '3D_irregular_small', '3D_regular_large', '3D_regular_medium', '3D_regular_small']
s3tsrs_pl = ['3D_irregular_small', '3D_regular_small', '3D_irregular_medium', '3D_regular_medium', '3D_irregular_large', '3D_regular_large']
s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
# s4tsrs_pl = ['4D_irregular_large', '4D_irregular_medium', '4D_irregular_small', '4D_regular_large', '4D_regular_medium', '4D_regular_small', '4D_i_large', '4D_i_medium', '4D_i_small']
# s4tsrs_pl = ['4D_irregular_large', '4D_irregular_medium', '4D_irregular_small', '4D_regular_large', '4D_regular_medium', '4D_regular_small']
# s4tsrs_pl = ['4D_i_large', '4D_i_medium', '4D_i_small', '4D_regular_large', '4D_regular_medium', '4D_regular_small']
s4tsrs_pl = ['4D_i_small', '4D_regular_small', '4D_i_medium', '4D_regular_medium', '4D_i_large', '4D_regular_large']

# For plots
s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'fb_s', 'deli', 'nell1']
# s3tsrs_pl_names =['irrL', 'irrM', 'irrS', 'regL', 'regM', 'regS']
s3tsrs_pl_names =['irrS', 'regS', 'irrM', 'regM', 'irrL', 'regL']
s4tsrs_names = ['crime4d', 'nips4d', 'enron4d', 'flickr4d', 'deli4d']
# s4tsrs_pl_names =['irrL4d', 'irrM4d', 'irrS4d', 'regL4d', 'regM4d', 'regS4d', 'irrL4d', 'irrM4d', 'irrS4d']
# s4tsrs_pl_names =['irrL4d', 'irrM4d', 'irrS4d', 'regL4d', 'regM4d', 'regS4d']
s4tsrs_pl_names =['irrS4d', 'regS4d', 'irrM4d', 'regM4d', 'irrL4d', 'regL4d']

# Global settings for figures
mywidth = 0.35      # the width of the bars

def main(argv):

	if len(argv) < 4:
		print("Usage: %s intput_path plot_tensors ang_pattern" % argv[0])
		exit(-1)

	# input parameters
	intput_path = sys.argv[1]
	plot_tensors = sys.argv[2]
	ang_pattern = sys.argv[3]
	print('intput_path: %s' % intput_path)
	print('plot_tensors: %s' % plot_tensors)
	print('ang_pattern: %s' % ang_pattern)

	if ang_pattern == '1':
		prefix = "dgx-2_"
		# prefix = "dgx-1_"
	else:
		prefix = ""

	if plot_tensors == "real":
		tensors = s3tsrs + s4tsrs
	elif plot_tensors == "graph":
		tensors = s3tsrs_pl + s4tsrs_pl
		# tensors = s3tsrs_pl 

	print(tensors)

	fig, (ax2, ax5) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

	nnzs = get_nnzs(tensors)

	gpu_gflops_coo = gpu_gflops_hicoo = []

	####### TEW #########
	# op = 'dadd_eq'
	# gpu_gflops_coo, gpu_gflops_hicoo = get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix)
	# rects1, rects2 = plot_gragh_left(ax1, plot_tensors, "TEW", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo))
	

	####### TS #########
	op = 'smul'
	gpu_gflops_coo, gpu_gflops_hicoo = get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix)
	# plot_gragh(ax2, plot_tensors, "TS", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo))
	rects1, rects2 = plot_gragh_left(ax2, plot_tensors, "TS", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo))

	####### TTV #########
	# op = 'ttv'
	# gpu_gflops_coo, gpu_gflops_hicoo = get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix)
	# plot_gragh(ax3, plot_tensors, "TTV", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo))
	
	####### TTM #########
	# op = 'ttm'
	# R = 16
	# gpu_gflops_coo, gpu_gflops_hicoo = get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, R, ang_pattern, prefix)
	# plot_gragh(ax4, plot_tensors, "TTM", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo))

	####### MTTKRP #########
	op = 'mttkrp'
	R = 16
	gpu_gflops_coo, gpu_gflops_hicoo = get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, R, ang_pattern, prefix)
	plot_gragh(ax5, plot_tensors, "MTTKRP", np.asarray(gpu_gflops_coo), np.asarray(gpu_gflops_hicoo))

	# fig.legend([], ['oral', 'physa'], bbox_to_anchor=(2, 0),loc = 'lower right')
	# fig.legend(*fig.axes[0,0].get_legend_handles_labels())

	fig.legend([rects1, rects2], ["gpu-coo", "gpu-hicoo"], loc = 'upper right') # bbox_to_anchor=(0.5, 0)

	# plt.show()
	plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')


def plot_gragh_left(ax, plot_tensors, title, o1, o2):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='gpu-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='gpu-hicoo')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind + mywidth)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects2


def plot_gragh(ax, plot_tensors, title, o1, o2):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='gpu-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='gpu-hicoo')

	ax.set_title(title, fontsize=20)
	ax.set_xticks(ind)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')


def get_nnzs(tensors):
	nnzs = []
	intput_path = '../timing-results-cori-save/'

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

	# print("nnzs:")
	# print(nnzs)

	return nnzs


def get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix):

	print("get_tew_data")
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
		if ang_pattern == '1':
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
		if ang_pattern == '1':
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

	print("gpu_times_coo:")
	print(gpu_times_coo)
	print("gpu_times_hicoo:")
	print(gpu_times_hicoo)

	# Calculate GFLOPS
	num_flops = nnzs
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	gpu_gflops_hicoo = [ float(num_flops[i]) / gpu_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("gpu_gflops_hicoo:")
	print(gpu_gflops_hicoo)
	print("\n")

	return gpu_gflops_coo, gpu_gflops_hicoo


def get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix):

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
		if ang_pattern == '1':
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
		if ang_pattern == '1':
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

	print("gpu_times_coo:")
	print(gpu_times_coo)
	print("gpu_times_hicoo:")
	print(gpu_times_hicoo)

	# Calculate GFLOPS
	num_flops = nnzs
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	gpu_gflops_hicoo = [ float(num_flops[i]) / gpu_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("gpu_gflops_hicoo:")
	print(gpu_gflops_hicoo)
	print("\n")

	return gpu_gflops_coo, gpu_gflops_hicoo


def get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, prefix):

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
			if ang_pattern == '1':
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
		gpu_times_coo.append(sum_time_modes)

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
			if ang_pattern == '1':
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
		gpu_times_hicoo.append(sum_time_modes)

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	print("gpu_times_coo:")
	print(gpu_times_coo)
	print("gpu_times_hicoo:")
	print(gpu_times_hicoo)

	# Calculate GFLOPS
	num_flops = [ 2 * i for i in nnzs ]
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	gpu_gflops_hicoo = [ float(num_flops[i]) / gpu_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("gpu_gflops_hicoo:")
	print(gpu_gflops_hicoo)
	print("\n")

	return gpu_gflops_coo, gpu_gflops_hicoo


def get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, R, ang_pattern, prefix):

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
			if ang_pattern == '1':
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
		gpu_times_coo.append(sum_time_modes)


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
			if ang_pattern == '1':
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
		# gpu_mode_times_hicoo.append(-1)
		# print("sum_time_modes:")
		# print(sum_time_modes)
		gpu_times_hicoo.append(sum_time_modes)

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	print("gpu_times_coo:")
	print(gpu_times_coo)
	print("gpu_times_hicoo:")
	print(gpu_times_hicoo)
	# print("gpu_mode_times_coo:")
	# print(gpu_mode_times_coo)
	print("gpu_mode_times_hicoo:")
	print(gpu_mode_times_hicoo)

	# Calculate GFLOPS
	num_flops = [ 2 * i * R for i in nnzs ]
	num_modes_flops = []
	for i in range(len(num_flops)):
		for m in range(nmodes):
			num_modes_flops.append(num_flops[i])
	print(num_modes_flops)
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	gpu_gflops_hicoo = [ float(num_flops[i]) / gpu_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	# gpu_mode_gflops_coo = [ float(num_modes_flops[i]) / gpu_mode_times_coo[i] / 1e9 for i in range(len(num_modes_flops)) ]
	gpu_mode_gflops_hicoo = [ float(num_modes_flops[i]) / gpu_mode_times_hicoo[i] / 1e9 for i in range(len(num_modes_flops)) ]
	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("gpu_gflops_hicoo:")
	print(gpu_gflops_hicoo)
	# print("gpu_mode_gflops_coo:")
	# print(gpu_mode_gflops_coo)
	print("gpu_mode_gflops_hicoo:")
	print(gpu_mode_gflops_hicoo)
	print("\n")

	return gpu_gflops_coo, gpu_gflops_hicoo
	# return gpu_mode_gflops_coo, gpu_mode_gflops_hicoo, theo_gflops_array


def get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, R, ang_pattern, prefix):

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
			if ang_pattern == '1':
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
		gpu_times_coo.append(sum_time_modes)


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
			if ang_pattern == '1':
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
		gpu_times_hicoo.append(sum_time_modes)

	assert(len(gpu_times_coo) == len(nnzs))
	assert(len(gpu_times_coo) == len(gpu_times_hicoo))

	print("gpu_times_coo:")
	print(gpu_times_coo)
	print("gpu_times_hicoo:")
	print(gpu_times_hicoo)

	# Calculate GFLOPS
	num_flops = [ 3 * i * R for i in nnzs ]
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	gpu_gflops_hicoo = [ float(num_flops[i]) / gpu_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("gpu_gflops_hicoo:")
	print(gpu_gflops_hicoo)
	print("\n")

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return gpu_gflops_coo, gpu_gflops_hicoo	


if __name__ == '__main__':
    sys.exit(main(sys.argv))


