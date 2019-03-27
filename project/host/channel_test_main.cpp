/////////////////////////////////////////
//
// OpenCL program template
// for testing the channels
// in nallatech p385a.
// Created by: Saman Biookaghazadeh @ ASU
//
////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>

#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"

#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR

using namespace std;
using namespace ocl_util;

// Define the kernel names used
const char *knl_name_tx = "sender";
const char *knl_name_rx = "receiver";

cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
cl_context context = NULL;
cl_program program_tx = NULL;
cl_program program_rx = NULL;

scoped_array<cl_device_id> device;
cl_kernel tx_kernel;
cl_kernel rx_kernel;

cl_command_queue que_tx;
cl_command_queue que_rx;

int main(int argc, char** argv) {

	cl_int status;
	int type = -1;
	cl_event tx_event;
	cl_event rx_event;

	if (argc != 4){
		printf("Error: wrong commad format, usage:\n");
		printf("%s <binaryfile>\n", argv[0]);
		return -1;
	}

	printf ("****************************\n");
	printf ("Nallatech P385A Channel Test\n");
	printf ("****************************\n");

	platform_id = findPlatform("Intel");
	if (platform_id == NULL) {
		printf ("ERROR: Unable to find the desired OpenCL platform.\n");
		return false;
	}

	device.reset(getDevices (platform_id, DEVICE_TYPE, &num_devices));
	if (num_devices != 2) 
		printf ("ERROR: there should be two devices installed on the system!\n");
	printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("  Device %d: %s\n", i, getDeviceName(device[i]).c_str());
		displayDeviceInfo(device[i]);
	}

	// Create the context.
	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");
	
	type = atoi(argv[1]);
	char* kernel_tx_file_name = argv[2];
	char* kernel_rx_file_name = argv[3];

	if (type != 1 && type != 2) {
		printf ("ERROR: type should be either 1 or 2!\n");
		return -1;
	}

	program_tx = createProgramFromFile(context, (const char *) kernel_tx_file_name, &(device[0]), 1);
	program_rx = createProgramFromFile(context, (const char *) kernel_rx_file_name, &(device[1]), 1);


	que_tx = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
	checkError (status, "Failed to create the tx command queue");
	que_rx = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
	checkError (status, "Failed to create the rx command queue");

	tx_kernel = clCreateKernel (program_tx, "sender", &status);
	checkError (status, "Failed to create the sender kernel");
	rx_kernel = clCreateKernel (program_rx, "receiver", &status);
	checkError (status, "Failed to create the receiver kernel");

	status = clEnqueueTask(que_tx, tx_kernel, 0, NULL, &tx_event);
	checkError (status, "Failed to enqueue the tx kernel");
	status = clEnqueueTask(que_rx, rx_kernel, 0, NULL, &rx_event);
	checkError (status, "Failed to enqueue the rx kernel");

	printf ("\nBoth kernels are pushed!\n");

	status = clWaitForEvents(1, &tx_event);
	checkError (status, "Failed to wait for the tx event!");
	status = clWaitForEvents(1, &rx_event);
	checkError (status, "Failed to wait for the rx event!");

	printf ("\nDone!\n");

}

// Release all memory resources here
void cleanup () {

	clReleaseKernel(rx_kernel);
	clReleaseKernel(tx_kernel);

	clReleaseCommandQueue(que_rx);
	clReleaseCommandQueue(que_tx);

	if (program_rx) {
		clReleaseProgram(program_rx);
	}

	if (program_tx) {
		clReleaseProgram(program_tx);
	}

	if (context) {
		clReleaseContext(context);
	}
}
