#!/usr/bin/python

import sys 

intput_path = '../timing-results/pasta/coo/'
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
operator=['dadd','ddiv','dmul','dsub','sadd','smul','ttv','mttkrp','ttm']
modes = ['0', '1', '2', '3']
op = 'mttkrp'
r = 16

# input parameters
tk = sys.argv[1]

out_str = str(op) + '-tk' + str(tk) + '.out'
print("Output file: " + out_str)
input_str = ""
fo = open(out_str, 'w')

for tsr in s4tsrs:
	sum_time_modes = 0.0

	for m in modes:

		if tk == "1":
			## sequential coo
			input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
		else:
			## omp coo
			input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(r) + '-t' + str(tk) + '.txt'
		print(input_str)

		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			if(len(line_array) < 4):
				continue;
			elif(line_array[1] == 'CooMTTKRP]:'):
				sum_time_modes = sum_time_modes + float(line_array[2])
 		fi.close()

	fo.write(str(sum_time_modes)+'\n')

fo.close()






