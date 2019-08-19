#!/usr/bin/env python

import sys
import subprocess

layers=57
if len(sys.argv) != 2:
	layer = -1
else:
	layer = int(sys.argv[1])

execute_layers_h = open("execute_layers.h", "w", 0)

def execute_layer(layer):
    execute_layers_h.seek(0)
    execute_layers_h.truncate(0)
    execute_layers_h.write(layer_configurations[layer])
    subprocess.call("~/pravin/GNU-DLA/project/generate.sh", shell=True)

with open('layer_configuration.txt') as layer_configuration:
    layer_configurations = layer_configuration.readlines()
  #   for x in xrange(0,layers):
		# print(layer_configurations[x])
    if layer == -1:
    	for x in xrange(0,layers):
    		execute_layer(int(x))
    else:
    	execute_layer(int(layer))