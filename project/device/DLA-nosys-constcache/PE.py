#!/usr/bin/env python

"""PE.py: The code generates the systolic array."""

__author__	= "Saman Biookaghazadeh"
__copyright__	= "Copyright 2019 @ Arizona State University"

var = input("Please enter the number of target processing elements: ")
target = open("PE.cl", "w")

with open('PE.cl.template') as pe_file:
	content = pe_file.read()

for id in range(var):

	content_pe_replaced = content.replace("$peid$", str(id))
	content_pep1_replaced = content_pe_replaced.replace("$peid+1$", str(id+1))

	target.write(content_pep1_replaced)
	target.write("\n\n")

target.close()
