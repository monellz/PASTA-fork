#!/usr/bin/python

import sys 

intput_path = './timing-results/pasta/coo/'
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
tmptsrs = ['vast-2015-mc1', 'choa700k', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
# operator=['dadd','ddiv','dmul','dsub']
op = 'ddiv_eq'

# input parameters
tk = sys.argv[1]

out_str = str(op) + '-tk' + str(tk) + '.out'
print("Output file: " + out_str)
input_str = ""
fo = open(out_str, 'w')

for tsr in s3tsrs + s4tsrs:
	sum_time = 0.0
	count = 0

	if tk == "1":
		## sequential coo TEW
		input_str = intput_path + tsr + '-' + op + '-c0' + '-seq.txt'
	else:
		## omp coo TEW
		input_str = intput_path + tsr + '-' + op + '-c0' + '-t' + str(tk) + '.txt'
	print(input_str)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		# print line_array
		if(len(line_array) < 4):
			continue;
		elif(line_array[2] == 'DotDiv]:'):
			count = count + 1
			if(count > 1):
				sum_time = sum_time + float(line_array[3])
				# print(sum_time)
	fi.close()
	time_num = sum_time / (count - 1)
	fo.write(str(time_num)+'\n')

fo.close()






