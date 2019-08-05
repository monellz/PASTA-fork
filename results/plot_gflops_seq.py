#!/usr/bin/python

import sys 
import numpy as np
import common

s3tsrs, s3tsrs_pl, s4tsrs, s4tsrs_pl, s3tsrs_names, s3tsrs_pl_names, s4tsrs_names, s4tsrs_pl_names = common.set_tsrnames()

# Global settings for figures
mywidth = 0.35      # the width of the bars

def main(argv):

	if len(argv) < 5:
		print("Usage: %s intput_path plot_tensors(real/graph) ang_pattern machine_name(cori, wingtip-bigmem2, dgx2, bluesky...)" % argv[0])
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

	# theoretical machine numbers
	if machine_name == "cori":
		theo_gflops = 1177.0
		theo_mem_bw = 102.0
		theo_cache_bw = 956.0
	elif machine_name == "wingtip-bigmem2":
		theo_gflops = 1971.0
		theo_mem_bw = 188.0
		theo_cache_bw = 1540.0
	elif machine_name == "bluesky":
		theo_gflops = 998.0
		theo_mem_bw = 190.0
		theo_cache_bw = 1183.0
	elif machine_name == "dgx2":
		theo_gflops = 1408.0
		theo_mem_bw = 79.0
		theo_cache_bw = 851.0
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

	# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(15, 3)) # 

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
	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw)
	# rects1, rects2, rects3 = plot_gragh_left(ax1, plot_tensors, "TEW", np.asarray(seq_gflops_coo), np.asarray(seq_gflops_hicoo), np.asarray(theo_gflops_array))
	# print("seq_gflops_coo average: %f" % np.average(seq_gflops_coo))
	# print("seq_gflops_hicoo average: %f" % np.average(seq_gflops_hicoo))
	# print("max gflops: %f" % max(max(seq_gflops_coo), max(seq_gflops_hicoo) ) )
	# print("")

	####### TS #########
	op = 'smul'
	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax2, plot_tensors, "TS", np.asarray(seq_gflops_coo), np.asarray(seq_gflops_hicoo), np.asarray(theo_gflops_array))
	# print("seq_gflops_coo average: %f" % np.average(seq_gflops_coo))
	# print("seq_gflops_hicoo average: %f" % np.average(seq_gflops_hicoo))
	# print("max gflops: %f" % max(max(seq_gflops_coo), max(seq_gflops_hicoo) ) )
	# print("\n")

	####### TTV #########
	op = 'ttv'
	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax3, plot_tensors, "TTV", np.asarray(seq_gflops_coo), np.asarray(seq_gflops_hicoo), np.asarray(theo_gflops_array))
	# print("seq_gflops_coo average: %f" % np.average(seq_gflops_coo))
	# print("seq_gflops_hicoo average: %f" % np.average(seq_gflops_hicoo))
	# print("max gflops: %f" % max(max(seq_gflops_coo), max(seq_gflops_hicoo) ) )
	# print("\n")

	####### TTM #########
	op = 'ttm'
	R = 16
	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, R, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax4, plot_tensors, "TTM", np.asarray(seq_gflops_coo), np.asarray(seq_gflops_hicoo), np.asarray(theo_gflops_array))
	# print("seq_gflops_coo average: %f" % np.average(seq_gflops_coo))
	# print("seq_gflops_hicoo average: %f" % np.average(seq_gflops_hicoo))
	# print("max gflops: %f" % max(max(seq_gflops_coo), max(seq_gflops_hicoo) ) )
	# print("\n")

	####### MTTKRP #########
	op = 'mttkrp'
	R = 16
	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, nbs, nnzbs, R, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw)
	# plot_gragh(ax5, plot_tensors, "MTTKRP", np.asarray(seq_gflops_coo), np.asarray(seq_gflops_hicoo), np.asarray(theo_gflops_array))
	# print("seq_gflops_coo average: %f" % np.average(seq_gflops_coo))
	# print("seq_gflops_hicoo average: %f" % np.average(seq_gflops_hicoo))
	# print("max gflops: %f" % max(max(seq_gflops_coo), max(seq_gflops_hicoo) ) )
	# print("\n")


	# fig.legend(loc = 'lower right', bbox_to_anchor=(2, 0), bbox_transform=ax1.transAxes)
	# fig.legend([rects1, rects2, rects3], ["omp-coo", "omp-hicoo", "roofline"], loc = 'upper right') # bbox_to_anchor=(0.5, 0)

	# plt.show()
	# plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')



def get_tew_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_tew_data")
	seq_times_coo = []
	seq_times_hicoo = []

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
			input_str = intput_path + tsr + '_' + op + '-seq.txt'
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
		seq_times_coo.append(time_num)

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
			input_str = intput_path + tsr + '_' + op + '_hicoo-b' + str(sb) + '-seq.txt'
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
		seq_times_hicoo.append(time_num)


	assert(len(seq_times_coo) == len(nnzs))
	assert(len(seq_times_coo) == len(seq_times_hicoo))

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

	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, seq_times_coo, seq_times_hicoo)

	# coo_gap_gflops = [ seq_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ seq_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_ts_data(op, intput_path, plot_tensors, tensors, nnzs, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ts_data")
	seq_times_coo = []
	seq_times_hicoo = []

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
			input_str = intput_path + tsr + '_' + op + '-seq.txt'
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
		seq_times_coo.append(time_num)

		###### HiCOO ######
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
			input_str = intput_path + tsr + '_' + op + '_hicoo-b' + str(sb) + '-seq.txt'
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
		seq_times_hicoo.append(time_num)

	assert(len(seq_times_coo) == len(nnzs))
	assert(len(seq_times_coo) == len(seq_times_hicoo))

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

	common.print_time_array(seq_times_coo, "times_coo")
	common.print_time_array(seq_times_hicoo, "times_hicoo")
	print("")
	sys.stdout.flush()

	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, seq_times_coo, seq_times_hicoo)

	# coo_gap_gflops = [ seq_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ seq_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_ttv_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ttv_data")
	seq_times_coo = []
	seq_times_hicoo = []

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
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-seq.txt'
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
		# print("time_modes coo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		seq_times_coo.append(avg_time_modes)	# could use min_time_modes


		###### HiCOO ######
		time_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		for m in modes:
			sum_time = 0.0
			count = 0
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo-m' + str(m) + '-b' + str(sb) + '-seq.txt'
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
		seq_times_hicoo.append(avg_time_modes)	# could use min_time_modes

	assert(len(seq_times_coo) == len(nnzs))
	assert(len(seq_times_coo) == len(seq_times_hicoo))

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
	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, seq_times_coo, seq_times_hicoo)

	# coo_gap_gflops = [ seq_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ seq_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_ttm_data(op, intput_path, plot_tensors, tensors, nnzs, nfibs, R, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_ttm_data")
	seq_times_coo = []
	seq_times_hicoo = []

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
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(R) + '-seq.txt'
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
		# print("time_modes coo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		seq_times_coo.append(avg_time_modes)	# could use min_time_modes


		###### HiCOO ######
		time_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		for m in modes:
			sum_time = 0.0
			count = 0
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-seq.txt'
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
		# print("time_modes hicoo:")
		# print(time_modes)
		avg_time_modes = sum(time_modes) / float(len(time_modes))
		min_time_modes = min(time_modes)
		seq_times_hicoo.append(avg_time_modes)	# could use min_time_modes

	assert(len(seq_times_coo) == len(nnzs))
	assert(len(seq_times_coo) == len(seq_times_hicoo))

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

	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, seq_times_coo, seq_times_hicoo)

	# coo_gap_gflops = [ seq_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ seq_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo


def get_mttkrp_data(op, intput_path, plot_tensors, tensors, nnzs, nbs, nnzbs, R, ang_pattern, theo_gflops, theo_mem_bw, theo_cache_bw):

	print("get_mttkrp_data")
	seq_times_coo = []
	seq_times_hicoo = []
	num_flops = []
	num_bytes_coo = []
	num_bytes_hicoo = []
	tsr_count = 0

	for tsr in tensors:
		# print(tsr)
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
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(R) + '-seq.txt'
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
		seq_times_coo.append(avg_time_modes)	# could use min_time_modes


		###### HiCOO ######
		time_modes = []
		if tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4
		else:
			sb = 7

		for m in modes:
			sum_time = 0.0
			count = 0
			## omp
			if ang_pattern == '1':
				input_str = intput_path + 'amd4_' + tsr + '_' + op + '_hicoo_Mode' + str(nmodes) + '_m' + str(m) + '_r16_omp-' + tk + '.txt'
			else:
				input_str = intput_path + tsr + '_' + op + '_hicoo-m' + str(m) + '-r' + str(R) + '-b' + str(sb) + '-seq.txt'
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
		seq_times_hicoo.append(avg_time_modes)	# could use min_time_modes

	assert(len(seq_times_coo) == len(nnzs))
	assert(len(seq_times_coo) == len(seq_times_hicoo))
	assert(tsr_count == len(nnzs))

	# Calculate GFLOPS and GBytes
	print("num_flops:")
	print(num_flops)
	print("num_bytes_coo:")
	print(num_bytes_coo)
	print("num_bytes_hicoo:")
	print(num_bytes_hicoo)
	print("")

	seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo = common.comp_gflops(theo_gflops, theo_mem_bw, theo_cache_bw, num_flops, num_bytes_coo, num_bytes_hicoo, seq_times_coo, seq_times_hicoo)

	# coo_gap_gflops = [ seq_gflops_coo[i] - seq_gflops_coo[i] for i in range(len(num_flops)) ]
	# hicoo_gap_gflops = [ seq_gflops_hicoo[i] - seq_gflops_hicoo[i] for i in range(len(num_flops)) ]

	return seq_gflops_coo, seq_gflops_hicoo, predicted_gflops_mem_coo, predicted_gflops_mem_hicoo, predicted_gflops_cache_coo, predicted_gflops_cache_hicoo

if __name__ == '__main__':
    sys.exit(main(sys.argv))


