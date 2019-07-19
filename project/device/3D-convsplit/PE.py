#!/usr/bin/env python

"""PE.py: The code generates the systolic array."""

__author__	= "Saman Biookaghazadeh"
__copyright__	= "Copyright 2019 @ Arizona State University"

import sys
if len(sys.argv) != 2:
	var = input("Please enter the number of target processing elements: ")
else:
	var = int(sys.argv[1])
target = open("PE.cl", "w")
target_output_channels = open("PE_header.cl", "w")
target_winograd = open("winograd_inv_transform.cl", "w")

with open('PE0.cl.template') as pe0_file:
	content0 = pe0_file.read()
with open('PEN.cl.template') as pen_file:
	contentn = pen_file.read()	
with open('PEF.cl.template') as pef_file:
	contentf = pef_file.read()
with open('winograd_inv_transform.cl.template') as win_file:
	contentwin = win_file.read()

for id in range(var):

	if id == 0:
		content0_pe_replaced = content0.replace("$peid$", str(id))
		content0_pep1_replaced = content0_pe_replaced.replace("$peid+1$", str(id+1))

		target.write(content0_pep1_replaced)
		target.write("\n\n")
	elif id != int(var)-1:
		contentn_pe_replaced = contentn.replace("$peid$", str(id))
		contentn_pep1_replaced = contentn_pe_replaced.replace("$peid+1$", str(id+1))
		contentn_pep_1_replaced = contentn_pep1_replaced.replace("$peid-1$", str(id-1))
		
		target.write(contentn_pep_1_replaced)
		target.write("\n\n")
	else:
		contentf_pe_replaced = contentf.replace("$peid$", str(id))
		contentf_pep1_replaced = contentf_pe_replaced.replace("$peid+1$", str(id+1))
		contentf_pep_1_replaced = contentf_pep1_replaced.replace("$peid-1$", str(id-1))
		
		target.write(contentf_pep_1_replaced)
		target.write("\n\n")

contentwin_replaced = contentwin.replace("$numpe-1$", str(int(var)-1))
target_winograd.write(contentwin_replaced)
target_winograd.write("\n\n")

target.close()
target_winograd.close()

for id in range(var):
	target_output_channels.write("typedef struct {\n")
	target_output_channels.write("\tw_data cols[")
	target_output_channels.write(str(id+1))
	target_output_channels.write("];\n")
	target_output_channels.write("} channel_cols")
	target_output_channels.write(str(id))
	target_output_channels.write(";\n\n")

for id in range(var):
	target_output_channels.write("channel channel_cols")
	target_output_channels.write(str(id))
	target_output_channels.write("\tchain_output_channels")
	target_output_channels.write(str(id))
	target_output_channels.write(";\n")

target_output_channels.write("\n")
target_output_channels.close()
