#!/usr/bin/env python

"""const_cache.py: The code generates const cache data, which contains the weight."""

import random

num_pes = input("Please enter the number of target processing elements: ")
output = open("const_cache.cl", "w")

for i in range(num_pes):

	output.write("__constant lane_cols weight_PE")
	output.write(str(i))
	output.write("[100] = {")
	for j in range(99):
		output.write(hex(random.randint(0, 256)))
		output.write(",")
	output.write(hex(random.randint(0, 256)))
	output.write("};\n")

	output.write("__constant DPTYPE bias_PE")
	output.write(str(i))
	output.write("[1] = {")
	output.write(hex(random.randint(0, 256)))
	output.write("};\n")

output.close()
