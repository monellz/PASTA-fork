#!/usr/bin/python

import sys 

intput_path = '../timing-results/pasta/coo/'
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
modes = ['0', '1', '2', '3']
op = 'ttm'
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
		sum_time = 0.0
		count = 0

		if tk == "1":
			## sequential coo TTV
			input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
		else:
			## omp coo TTV
			input_str = intput_path + tsr + '_' + op + '-m' + str(m) + '-r' + str(r) + '-t' + str(tk) + '.txt'
		print(input_str)

		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			if(len(line_array) < 6):
				continue;
			elif(line_array[4] == 'Mtx]:'):
				count = count + 1
				if(count > 1):
					sum_time = sum_time + float(line_array[5])
 		fi.close()
 		time_num = sum_time / (count - 1)
 		sum_time_modes = sum_time_modes + time_num

 	fo.write(str(sum_time_modes)+'\n')

fo.close()






