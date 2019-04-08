#!/usr/bin/python

import sys 
import matplotlib.pyplot as plt
import numpy as np

# For locating data
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
s3tsrs_pl = ['3D_irregular_large', '3D_irregular_medium', '3D_irregular_small', '3D_regular_large', '3D_regular_medium', '3D_regular_small']
s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
# s4tsrs_pl = ['4D_irregular_large', '4D_irregular_medium', '4D_irregular_small', '4D_regular_large', '4D_regular_medium', '4D_regular_small', '4D_i_large', '4D_i_medium', '4D_i_small']
s4tsrs_pl = ['4D_irregular_large', '4D_irregular_medium', '4D_irregular_small', '4D_regular_large', '4D_regular_medium', '4D_regular_small']

# For plots
s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'fb_s', 'deli', 'nell1']
s3tsrs_pl_names =['irrL', 'irrM', 'irrS', 'regL', 'regM', 'regS']
s4tsrs_names = ['crime4d', 'nips4d', 'enron4d', 'flickr4d', 'deli4d']
# s4tsrs_pl_names =['irrL4d', 'irrM4d', 'irrS4d', 'regL4d', 'regM4d', 'regS4d', 'irrL4d', 'irrM4d', 'irrS4d']
s4tsrs_pl_names =['irrL4d', 'irrM4d', 'irrS4d', 'regL4d', 'regM4d', 'regS4d']

# gflops from roofline model
theo_gflops_tew = 10
theo_gflops_ts = 10
theo_gflops_ttv = 10
theo_gflops_ttm = 10
theo_gflops_mttkrp = 10

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
		# prefix = "dgx-2_"
		prefix = "dgx-1_"
	else:
		prefix = ""

	if plot_tensors == "real":
		tensors = s3tsrs + s4tsrs
	elif plot_tensors == "graph":
		tensors = s3tsrs_pl + s4tsrs_pl

	print(tensors)

	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(100, 3))

	nnzs = get_nnzs(tensors)

	gpu_gflops_coo = theo_gflops_array = []

	####### TEW #########
	op = 'dadd_eq'
	gpu_gflops_coo, theo_gflops_array = get_tew_data(op, intput_path, theo_gflops_tew, plot_tensors, tensors, nnzs, ang_pattern, prefix)
	rects1, rects3 = plot_gragh_left(ax1, plot_tensors, "TEW", np.asarray(gpu_gflops_coo), np.asarray(theo_gflops_array))
	
	fig.legend([rects1, rects3], ["gpu-coo", "roofline"], loc = 'upper right') # bbox_to_anchor=(0.5, 0)

	####### TS #########
	op = 'smul'
	gpu_gflops_coo, theo_gflops_array = get_ts_data(op, intput_path, theo_gflops_ts, plot_tensors, tensors, nnzs, ang_pattern, prefix)
	plot_gragh(ax2, plot_tensors, "TS", np.asarray(gpu_gflops_coo), np.asarray(theo_gflops_array))

	####### TTV #########
	op = 'ttv'
	gpu_gflops_coo, theo_gflops_array = get_ttv_data(op, intput_path, theo_gflops_ttv, plot_tensors, tensors, nnzs, ang_pattern, prefix)
	plot_gragh(ax3, plot_tensors, "TTV", np.asarray(gpu_gflops_coo), np.asarray(theo_gflops_array))
	
	####### TTM #########
	op = 'ttm'
	R = 16
	gpu_gflops_coo, theo_gflops_array = get_ttm_data(op, intput_path, theo_gflops_ttm, plot_tensors, tensors, nnzs, R, ang_pattern, prefix)
	plot_gragh(ax4, plot_tensors, "TTM", np.asarray(gpu_gflops_coo), np.asarray(theo_gflops_array))

	####### MTTKRP #########
	op = 'mttkrp'
	R = 16
	gpu_gflops_coo, theo_gflops_array = get_mttkrp_data(op, intput_path, theo_gflops_mttkrp, plot_tensors, tensors, nnzs, R, ang_pattern, prefix)
	plot_gragh(ax5, plot_tensors, "MTTKRP", np.asarray(gpu_gflops_coo), np.asarray(theo_gflops_array))

	# fig.legend([], ['oral', 'physa'], bbox_to_anchor=(2, 0),loc = 'lower right')
	# fig.legend(*fig.axes[0,0].get_legend_handles_labels())


	plt.show()


def plot_gragh_left(ax, plot_tensors, title, o1, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='m', zorder=2, lw=0.5, label='gpu-coo')
	rects3 = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects3


def plot_gragh(ax, plot_tensors, title, o1, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='m', zorder=2, lw=0.5, label='gpu-coo')
	rects3 = ax.plot(ind , o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_xticks(ind)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')


def get_nnzs(tensors):
	nnzs = []
	intput_path = '../timing-results-cori/'

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

def get_tew_data(op, intput_path, theo_gflops, plot_tensors, tensors, nnzs, ang_pattern, prefix):

	print("get_tew_data")
	gpu_times_coo = []

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


	assert(len(gpu_times_coo) == len(nnzs))

	print("gpu_times_coo:")
	print(gpu_times_coo)

	# Calculate GFLOPS
	num_flops = nnzs
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]

	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	return gpu_gflops_coo, theo_gflops_array


def get_ts_data(op, intput_path, theo_gflops, plot_tensors, tensors, nnzs, ang_pattern, prefix):

	print("get_ts_data")
	gpu_times_coo = []

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

	assert(len(gpu_times_coo) == len(nnzs))

	print("gpu_times_coo:")
	print(gpu_times_coo)

	# Calculate GFLOPS
	num_flops = nnzs
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]

	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	return gpu_gflops_coo, theo_gflops_array


def get_ttv_data(op, intput_path, theo_gflops, plot_tensors, tensors, nnzs, ang_pattern, prefix):

	print("get_ttv_data")
	gpu_times_coo = []

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
		gpu_times_coo.append(time_num)

	assert(len(gpu_times_coo) == len(nnzs))

	print("gpu_times_coo:")
	print(gpu_times_coo)

	# Calculate GFLOPS
	num_flops = [ 2 * i for i in nnzs ]
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]

	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	return gpu_gflops_coo, theo_gflops_array


def get_ttm_data(op, intput_path, theo_gflops, plot_tensors, tensors, nnzs, R, ang_pattern, prefix):

	print("get_ttm_data")
	gpu_times_coo = []

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
			sum_time_modes += time_num
			# print(time_num)
		sum_time_modes /= nmodes
		# print(sum_time_modes)
		gpu_times_coo.append(time_num)

	assert(len(gpu_times_coo) == len(nnzs))

	print("gpu_times_coo:")
	print(gpu_times_coo)

	# Calculate GFLOPS
	num_flops = [ 2 * i * R for i in nnzs ]
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]

	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	return gpu_gflops_coo, theo_gflops_array


def get_mttkrp_data(op, intput_path, theo_gflops, plot_tensors, tensors, nnzs, R, ang_pattern, prefix):

	print("get_mttkrp_data")
	gpu_times_coo = []

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
		gpu_times_coo.append(time_num)

	assert(len(gpu_times_coo) == len(nnzs))

	print("gpu_times_coo:")
	print(gpu_times_coo)

	# Calculate GFLOPS
	num_flops = [ 3 * i * R for i in nnzs ]
	gpu_gflops_coo = [ float(num_flops[i]) / gpu_times_coo[i] / 1e9 for i in range(len(num_flops)) ]

	print("num_flops:")
	print(num_flops)
	print("gpu_gflops_coo:")
	print(gpu_gflops_coo)
	print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	return gpu_gflops_coo, theo_gflops_array	


if __name__ == '__main__':
    sys.exit(main(sys.argv))


