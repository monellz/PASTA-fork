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

# gflops from roofline model

# Cori: ERT BW: 102
# theo_gflops_tew = 8.5
# theo_gflops_ts = 12.75
# theo_gflops_ttv = 25.5
# theo_gflops_ttm = 51
# theo_gflops_mttkrp = 25.5

# wingtip: ERT BW: 188
# theo_gflops_tew = 15.69
# theo_gflops_ts = 23.54
# theo_gflops_ttv = 47.07
# theo_gflops_ttm = 94.145
# theo_gflops_mttkrp = 47.07

# dgx-cpu: ERT BW: 79
# theo_gflops_tew = 6.58333333333333333333
# theo_gflops_ts = 9.875
# theo_gflops_ttv = 19.75
# theo_gflops_ttm = 39.5
# theo_gflops_mttkrp = 19.75

# bluesky: ERT BW: 190
theo_gflops_tew = 15.81583333333333333333
theo_gflops_ts = 23.72375
theo_gflops_ttv = 47.4475
theo_gflops_ttm = 94.145
theo_gflops_mttkrp = 47.4475



# Global settings for figures
mywidth = 0.35      # the width of the bars

def main(argv):

	if len(argv) < 5:
		print("Usage: %s intput_path tk plot_tensors ang_pattern" % argv[0])
		exit(-1)

	# input parameters
	intput_path = sys.argv[1]
	tk = sys.argv[2]
	plot_tensors = sys.argv[3]
	ang_pattern = sys.argv[4]
	print('intput_path: %s' % intput_path)
	print('tk: %s' % tk)
	print('plot_tensors: %s' % plot_tensors)
	print('ang_pattern: %s' % ang_pattern)

	if plot_tensors == "real":
		tensors = s3tsrs + s4tsrs
	elif plot_tensors == "graph":
		tensors = s3tsrs_pl + s4tsrs_pl

	print(tensors)

	fig, (ax2, ax5) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3)) # 

	nnzs = get_nnzs(tensors)

	omp_gflops_coo = omp_gflops_hicoo = theo_gflops_array = []

	####### TEW #########
	# op = 'dadd_eq'
	# omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array = get_tew_data(op, intput_path, tk, theo_gflops_tew, plot_tensors, tensors, nnzs, ang_pattern)
	# rects1, rects2 = plot_gragh_left(ax1, plot_tensors, "TEW", np.asarray(omp_gflops_coo), np.asarray(omp_gflops_hicoo))
	# print("omp_gflops_coo average: %f" % np.average(omp_gflops_coo))
	# print("omp_gflops_hicoo average: %f" % np.average(omp_gflops_hicoo))
	# print("max gflops: %f" % max(max(omp_gflops_coo), max(omp_gflops_hicoo) ) )
	# print("\n")

	####### TS #########
	op = 'smul'
	omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array = get_ts_data(op, intput_path, tk, theo_gflops_ts, plot_tensors, tensors, nnzs, ang_pattern)
	# plot_gragh(ax2, plot_tensors, "TS", np.asarray(omp_gflops_coo), np.asarray(omp_gflops_hicoo))
	rects1, rects2 = plot_gragh_left(ax2, plot_tensors, "TS", np.asarray(omp_gflops_coo), np.asarray(omp_gflops_hicoo))
	print("omp_gflops_coo average: %f" % np.average(omp_gflops_coo))
	print("omp_gflops_hicoo average: %f" % np.average(omp_gflops_hicoo))
	print("max gflops: %f" % max(max(omp_gflops_coo), max(omp_gflops_hicoo) ) )
	print("\n")

	####### TTV #########
	# op = 'ttv'
	# omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array = get_ttv_data(op, intput_path, tk, theo_gflops_ttv, plot_tensors, tensors, nnzs, ang_pattern)
	# plot_gragh(ax3, plot_tensors, "TTV", np.asarray(omp_gflops_coo), np.asarray(omp_gflops_hicoo))
	# print("omp_gflops_coo average: %f" % np.average(omp_gflops_coo))
	# print("omp_gflops_hicoo average: %f" % np.average(omp_gflops_hicoo))
	# print("max gflops: %f" % max(max(omp_gflops_coo), max(omp_gflops_hicoo) ) )
	# print("\n")

	####### TTM #########
	# op = 'ttm'
	# R = 16
	# omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array = get_ttm_data(op, intput_path, tk, theo_gflops_ttm, plot_tensors, tensors, nnzs, R, ang_pattern)
	# plot_gragh(ax4, plot_tensors, "TTM", np.asarray(omp_gflops_coo), np.asarray(omp_gflops_hicoo))
	# print("omp_gflops_coo average: %f" % np.average(omp_gflops_coo))
	# print("omp_gflops_hicoo average: %f" % np.average(omp_gflops_hicoo))
	# print("max gflops: %f" % max(max(omp_gflops_coo), max(omp_gflops_hicoo) ) )
	# print("\n")

	####### MTTKRP #########
	op = 'mttkrp'
	R = 16
	omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array = get_mttkrp_data(op, intput_path, tk, theo_gflops_mttkrp, plot_tensors, tensors, nnzs, R, ang_pattern)
	plot_gragh(ax5, plot_tensors, "MTTKRP", np.asarray(omp_gflops_coo), np.asarray(omp_gflops_hicoo))
	print("omp_gflops_coo average: %f" % np.average(omp_gflops_coo))
	print("omp_gflops_hicoo average: %f" % np.average(omp_gflops_hicoo))
	print("max gflops: %f" % max(max(omp_gflops_coo), max(omp_gflops_hicoo) ) )
	print("\n")


	# fig.legend(loc = 'lower right', bbox_to_anchor=(2, 0), bbox_transform=ax1.transAxes)
	fig.legend([rects1, rects2], ["omp-coo", "omp-hicoo"], loc = 'upper right') # bbox_to_anchor=(0.5, 0)


	# plt.show()
	plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')


def plot_gragh_left(ax, plot_tensors, title, o1, o2):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='omp-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m', zorder=2, lw=0.5, label='omp-hicoo')
	# rects3, = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind + mywidth)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2)) + ylim_var] )

	# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('seq-coo', 'omp-coo','seq-hicoo','omp-hicoo', 'roofline'),loc='right', shadow=True)
	# ax.legend()
	ax.grid(axis='y')
	# ax.autoscale_view()

	# for rect in rects1:
	#     height = rect.get_height()
	#     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects2


def plot_gragh(ax, plot_tensors, title, o1, o2):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='omp-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m', zorder=2, lw=0.5, label='omp-hicoo')
	# rects3, = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_xticks(ind + mywidth)
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

def get_tew_data(op, intput_path, tk, theo_gflops, plot_tensors, tensors, nnzs, ang_pattern):

	print("get_tew_data")
	omp_times_coo = []
	omp_times_hicoo = []

	for tsr in tensors:
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4

		###### COO ######
		sum_time = 0.0
		count = 0
		## omp
		if ang_pattern == '1':
			input_str = intput_path + 'amd4_' + tsr + '_' + op + '_Mode' + str(nmodes) + '_omp-' + tk + '.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '-t' + tk + '.txt'
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
		omp_times_coo.append(time_num)

		###### HiCOO ######
		# if tsr in s4tsrs:
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time = 0.0
		count = 0
		## omp
		if ang_pattern == '1':
			input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_omp-' + tk + '.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '_hicoo-b' + str(sb) + '-t' + tk + '.txt'
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
		omp_times_hicoo.append(time_num)


	assert(len(omp_times_coo) == len(nnzs))
	assert(len(omp_times_coo) == len(omp_times_hicoo))

	# print("omp_times_coo:")
	# print(omp_times_coo)
	# print("omp_times_hicoo:")
	# print(omp_times_hicoo)

	# Calculate GFLOPS
	num_flops = nnzs
	omp_gflops_coo = [ float(num_flops[i]) / omp_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	omp_gflops_hicoo = [ float(num_flops[i]) / omp_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	# print("num_flops:")
	# print(num_flops)
	# print("omp_gflops_coo:")
	# print(omp_gflops_coo)
	# print("omp_gflops_hicoo:")
	# print(omp_gflops_hicoo)
	# print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array


def get_ts_data(op, intput_path, tk, theo_gflops, plot_tensors, tensors, nnzs, ang_pattern):

	print("get_ts_data")
	omp_times_coo = []
	omp_times_hicoo = []

	for tsr in tensors:
		if tsr in s3tsrs + s3tsrs_pl:
			nmodes = 3
		elif tsr in s4tsrs + s4tsrs_pl:
			nmodes = 4

		###### COO ######
		sum_time = 0.0
		count = 0
		## omp
		if ang_pattern == '1':
			input_str = intput_path + 'amd4_' + tsr + '_' + op + '_Mode' + str(nmodes) + '_omp-' + tk + '.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '-t' + tk + '.txt'
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
		omp_times_coo.append(time_num)

		###### HiCOO ######
		# if tsr in s4tsrs:	# for cori data
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time = 0.0
		count = 0
		## omp
		if ang_pattern == '1':
			input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_omp-' + tk + '.txt'
		else:
			input_str = intput_path + tsr + '_' + op + '_hicoo-b' + str(sb) + '-t' + tk + '.txt'
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
		omp_times_hicoo.append(time_num)

	assert(len(omp_times_coo) == len(nnzs))
	assert(len(omp_times_coo) == len(omp_times_hicoo))

	# print("omp_times_coo:")
	# print(omp_times_coo)
	# print("omp_times_hicoo:")
	# print(omp_times_hicoo)

	# Calculate GFLOPS
	num_flops = nnzs
	omp_gflops_coo = [ float(num_flops[i]) / omp_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	omp_gflops_hicoo = [ float(num_flops[i]) / omp_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	print("num_flops:")
	print(num_flops)
	# print("omp_gflops_coo:")
	# print(omp_gflops_coo)
	# print("omp_gflops_hicoo:")
	# print(omp_gflops_hicoo)
	# print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array


def get_ttv_data(op, intput_path, tk, theo_gflops, plot_tensors, tensors, nnzs, ang_pattern):

	print("get_ttv_data")
	omp_times_coo = []
	omp_times_hicoo = []

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
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-t' + tk + '.txt'
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
		omp_times_coo.append(sum_time_modes)

		###### HiCOO ######
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time_modes = 0.0
		for m in modes:
			sum_time = 0.0
			count = 0
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo-m' + str(m) + '-b' + str(sb) + '-t' + tk + '.txt'
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
		omp_times_hicoo.append(sum_time_modes)

	assert(len(omp_times_coo) == len(nnzs))
	assert(len(omp_times_coo) == len(omp_times_hicoo))

	# print("omp_times_coo:")
	# print(omp_times_coo)
	# print("omp_times_hicoo:")
	# print(omp_times_hicoo)
	

	# Calculate GFLOPS
	num_flops = [ 2 * i for i in nnzs ]
	omp_gflops_coo = [ float(num_flops[i]) / omp_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	omp_gflops_hicoo = [ float(num_flops[i]) / omp_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	# print("num_flops:")
	# print(num_flops)
	# print("omp_gflops_coo:")
	# print(omp_gflops_coo)
	# print("omp_gflops_hicoo:")
	# print(omp_gflops_hicoo)
	# print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array


def get_ttm_data(op, intput_path, tk, theo_gflops, plot_tensors, tensors, nnzs, R, ang_pattern):

	print("get_ttm_data")
	omp_times_coo = []
	omp_times_hicoo = []

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
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(R) + '-t' + tk + '.txt'
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
		omp_times_coo.append(sum_time_modes)

		###### HiCOO ######
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time_modes = 0.0
		for m in modes:
			sum_time = 0.0
			count = 0
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-t' + tk + '.txt'
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
		# print("sum_time_modes:")
		# print(sum_time_modes)
		omp_times_hicoo.append(sum_time_modes)

	assert(len(omp_times_coo) == len(nnzs))
	assert(len(omp_times_coo) == len(omp_times_hicoo))

	# print("omp_times_coo:")
	# print(omp_times_coo)
	# print("omp_times_hicoo:")
	# print(omp_times_hicoo)

	# Calculate GFLOPS
	num_flops = [ 2 * i * R for i in nnzs ]
	omp_gflops_coo = [ float(num_flops[i]) / omp_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	omp_gflops_hicoo = [ float(num_flops[i]) / omp_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	# print("num_flops:")
	# print(num_flops)
	# print("omp_gflops_coo:")
	# print(omp_gflops_coo)
	# print("omp_gflops_hicoo:")
	# print(omp_gflops_hicoo)
	# print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array


def get_mttkrp_data(op, intput_path, tk, theo_gflops, plot_tensors, tensors, nnzs, R, ang_pattern):

	print("get_mttkrp_data")
	omp_times_coo = []
	omp_times_hicoo = []

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
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(R) + '-t' + tk + '.txt'
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
		omp_times_coo.append(sum_time_modes)

		###### HiCOO ######
		# if tsr in ["chicago-crime-comm-4d", "uber-4d", "enron-4d", "nips-4d"]:	# for cori data
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		sum_time_modes = 0.0
		for m in modes:
			sum_time = 0.0
			count = 0
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-t' + tk + '.txt'
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
		omp_times_hicoo.append(sum_time_modes)

	assert(len(omp_times_coo) == len(nnzs))
	assert(len(omp_times_coo) == len(omp_times_hicoo))

	# print("omp_times_coo:")
	# print(omp_times_coo)
	# print("omp_times_hicoo:")
	# print(omp_times_hicoo)

	# Calculate GFLOPS
	num_flops = [ 3 * i * R for i in nnzs ]
	omp_gflops_coo = [ float(num_flops[i]) / omp_times_coo[i] / 1e9 for i in range(len(num_flops)) ]
	omp_gflops_hicoo = [ float(num_flops[i]) / omp_times_hicoo[i] / 1e9 for i in range(len(num_flops)) ]
	# print("num_flops:")
	# print(num_flops)
	# print("omp_gflops_coo:")
	# print(omp_gflops_coo)
	# print("omp_gflops_hicoo:")
	# print(omp_gflops_hicoo)
	# print("\n")

	theo_gflops_array = [theo_gflops] * len(num_flops)

	# coo_gap_gflops = [ omp_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ omp_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return omp_gflops_coo, omp_gflops_hicoo, theo_gflops_array	

if __name__ == '__main__':
    sys.exit(main(sys.argv))


