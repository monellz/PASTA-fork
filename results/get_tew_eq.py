#!/usr/bin/python

import sys 
import matplotlib.pyplot as plt
import numpy as np

# For locating data
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
s3tsrs_pl = ['3D_irregular_large', '3D_irregular_medium', '3D_irregular_small', '3D_regular_large', '3D_regular_medium', '3D_regular_small']
s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
s4tsrs_pl = ['4D_irregular_large', '4D_irregular_medium', '4D_irregular_small', '4D_regular_large', '4D_regular_medium', '4D_regular_small', '4D_i_large', '4D_i_medium', '4D_i_small']

# For plots
s3tsrs_names = ['vast', 'nell2', 'choa', 'darpa', 'fb_m', 'fb_s', 'deli', 'nell1']
s3tsrs_pl_names =['irr_l', 'irr_m', 'irr_s', 'reg_l', 'reg_m', 'reg_s']
s4tsrs_names = ['crime_4d', 'nips_4d', 'enron_4d', 'flickr_4d', 'deli_4d']
s4tsrs_pl_names =['irr_l_4d', 'irr_m_4d', 'irr_s_4d', 'reg_l_4d', 'reg_m_4d', 'reg_s_4d', 'irr_l_4d', 'irr_m_4d', 'irr_s_4d']

def plot_gragh(ax, plot_tensors, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind= np.arange(len(o1))
	mywidth = 0.35      # the width of the bars
	ylim_var = 0.2

	rects1 = ax.bar(left=ind, height=o2, width=mywidth, color='b', label='seq')
	rects2 = ax.bar(left=[ i + mywidth for i in ind ], height=o1, width=mywidth, color='g', label='omp')
	rects3 = ax.plot(ind, o3, color='r', label='roofline')
	ax.set_ylim(ymax=o3[0]+ylim_var)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=18)
	ax.set_xticks(ind)
	ax.set_xticklabels(xnames, fontsize=16, rotation=90)


def main(argv):
	op = 'dadd_eq'
	
	if len(argv) < 6:
		print("Usage: %s intput_path tk theo_gflops plot_tensors tensor_format" % argv[0])
		exit(-1)

	# input parameters
	intput_path = sys.argv[1]
	tk = sys.argv[2]
	theo_gflops = float(sys.argv[3])
	plot_tensors = sys.argv[4]
	tsr_format = sys.argv[5]

	seq_times = []
	omp_times = []
	nnzs = []

	if plot_tensors == "real":
		tensors = s3tsrs + s4tsrs
	elif plot_tensors == "graph":
		tensors = s3tsrs_pl + s4tsrs_pl

	for tsr in tensors:

		sb = 7
		if tsr_format == "hicoo" and tsr in ["chicago-crime-comm-4d", "uber-4d"]:
			sb = 4

		sum_time = 0.0
		count = 0
		## sequential 
		if tsr_format == "coo":
			input_str = intput_path + tsr + '_' + op + '-seq.txt'
		elif tsr_format == "hicoo":
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
		seq_times.append(time_num)

		sum_time = 0.0
		count = 0
		## omp 
		if tsr_format == "coo":
			input_str = intput_path + tsr + '_' + op + '-t' + tk + '.txt'
		elif tsr_format == "hicoo":
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
		omp_times.append(time_num)

		count = 0
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

	assert(len(seq_times) == len(omp_times))
	assert(len(seq_times) == len(nnzs))

	# print("seq_times:")
	# print(seq_times)
	# print("omp_times:")
	# print(omp_times)
	# print("nnzs:")
	# print(nnzs)

	# Calculate GFLOPS
	num_flops = nnzs
	seq_gflops = [ float(num_flops[i]) / seq_times[i] / 1e9 for i in range(len(num_flops)) ]
	omp_gflops = [ float(num_flops[i]) / omp_times[i] / 1e9 for i in range(len(num_flops)) ]
	# print("num_flops:")
	# print(num_flops)
	print("seq_gflops:")
	print(seq_gflops)
	print("omp_gflops:")
	print(omp_gflops)

	theo_gflops_array = [theo_gflops] * len(seq_times)

	fig, (ax0) = plt.subplots(nrows=1, ncols=1)

	plot_gragh(ax0, plot_tensors, seq_gflops, omp_gflops, theo_gflops_array)

	plt.show()



if __name__ == '__main__':
    sys.exit(main(sys.argv))


