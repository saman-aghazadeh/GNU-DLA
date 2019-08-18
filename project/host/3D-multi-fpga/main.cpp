//////////////////////////////////////////
//
// OpenCL host program template for multiple
// FPGA boards.
//
// Created by dongwang@2016.01.10
//
/////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <chrono>

// Define colors
#define ANSI_COLOR_RED		"\x1b[31m"
#define ANSI_COLOR_GREEN	"\x1b[32m"
#define ANSI_COLOR_YELLOW	"\x1b[33m"
#define ANSI_COLOR_BLUE		"\x1b[34m"
#define ANSI_COLOR_MAGENTA	"\x1b[35m"
#define ANSI_COLOR_CYAN		"\x1b[36m"
#define ANSI_COLOR_RESET	"\x1b[0m"

#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"

// CNN network configuration file
#include "../../device/DLA-nosys/hw_param.cl"
#include "layer_config.h"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
#endif

using namespace std;
using namespace ocl_util;

typedef  signed char  DTYPE;

#ifdef XILINX
//#define USE_SDX_1DDR  // reserved for v7-690T device, DO NOT USE
//#define USE_SDX_4DDR  // reverved for 4-bank DDR BSP, DO NOT USE
#endif

//----------- Design Parameters --------------//
// select what platform is used
const char *vendor_name = "Intel";
#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR

// SW System parameters
#define DMA_ALIGNMENT   64

#define IN_BUF_SIZE    256*256*64  // Note: the buffer size should be large enough to hold all temperary results
#define OUT_BUF_SIZE   256*256*64

// #define VERBOSE_OUTPUT

#ifdef ALEXNET_TEST
// Original problem size
// File size is in num of DTYPE numbers
#define IMAGE_FILE_SIZE   (227*227*3)
//#define WEIGHTS_FILE_SIZE 60965224 //fc8-1000
#define WEIGHTS_FILE_SIZE 61063552  //fc8-1024
#define LAYER_NUM         8
#define CONV_NUM          5
const char *weight_file_path = "./data/data_alex/weights.dat";
const char *input_file_path = "./data/data_alex/image.dat";
#endif

#ifdef VGG16_TEST
// VGG16
// Original problem size
// File size is in num of DTYPE numbers
#define IMAGE_FILE_SIZE   (224*224*3)
#define WEIGHTS_FILE_SIZE 324442112  //fc8-1024
#define BIASES_FILE_SIZE  13440
#define LAYER_NUM         16
#define CONV_NUM          16
const char *weight_file_path = "./data/data_vgg16/weights.dat";
const char *input_file_path = "./data/data_vgg16/image.dat";
#endif

#ifdef C3D_TEST
#define IMAGE_FILE_SIZE		128*171*3*16
#define WEIGHTS_FILE_SIZE	291879616
#define BIASES_FILE_SIZE	10944
#define LAYER_NUM		10
#define CONV_NUM		8
#define IN_BUF_SIZE		55705600
#define OUT_BUF_SIZE		55705600
const char *weight_file_path = "./data/data_vgg16/weights.dat";
const char *input_file_path = "./data/data_vgg16/image.dat";	
#endif

// Configuration file instructions
enum config_item{
layer_type, // "0" -> conv, "1" -> fc

data_w, data_h, data_n, data_t, weight_w, weight_h, weight_n, weight_t, weight_m, bias_size, //memRd Parameters

memrd_src, //"0"-> data_buf  "1"-> output_buf  "2"->"fc_1_buffer"  "3"->"fc_2_buffer"

conv_x, conv_y, conv_z, conv_t, conv_stride, conv_padding, conv_split, conv_relu, //Conv Parameters

pool_on, pool_x, pool_y, pool_z, pool_t, pool_size_xy, pool_size_t, pool_stride_xy, pool_stride_t, // Pooling Parameters

lrn_on,// lrn on/off control

memwr_dst//"0"-> data_buf  "1"-> output_buf  "2"->"fc_1_buffer"  "3"->"fc_2_buffer"
};

enum input_item{
image_w, image_h, image_n, image_t,// original image size
batch_size
};

enum output_item{
output_w, output_h, output_n
};

enum precision_item{
frac_w, frac_din, frac_dout
};

typedef struct {
	int layer_type;
	int data_w, data_h, data_t, weight_w, weight_h, weight_n, weight_t, weight_m, bias_size;
	int memrd_src;
	int conv_x, conv_y, conv_z, conv_t, conv_stride, conv_padding, conv_split, conv_relu;
	int pool_on, pool_x, pool_y, pool_z, pool_t, pool_size_xy, pool_size_t, pool_stride_xy, pool_stride_t;
	int lrn_on;
	int memwr_dst;
	int num_bricks;
} fpga_configuration;

typedef struct device_runner_arg {
	int device;
} device_runner_arg;

// Define the kernel names used
const char *knl_name_memRdData = "memReadData";
const char *knl_name_memRdWeight = "memReadWeight";
const char *knl_name_controller = "controller";
const char *knl_name_memWrite = "memWrite";
const char *knl_name_ser = "ser";
const char *knl_name_deser = "deser";

//------------ Global Functions & Variables ------------//
cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
scoped_array<cl_context> context;
scoped_array<cl_program> program;
scoped_array<cl_device_id> device;
scoped_array<cl_kernel> knl_memRdData;
scoped_array<cl_kernel> knl_memRdWeight;
scoped_array<cl_kernel> knl_controller;
scoped_array<cl_kernel> knl_memWrite;
scoped_array<cl_kernel> knl_ser;
scoped_array<cl_kernel> knl_deser;

scoped_array<cl_command_queue> que_memRdData;
scoped_array<cl_command_queue> que_memRdWeight;
scoped_array<cl_command_queue> que_controller;
scoped_array<cl_command_queue> que_memWrite;

scoped_array<cl_mem> config_buf;
scoped_array<cl_mem> bottom0_buf;
scoped_array<cl_mem> bottom1_buf;
scoped_array<cl_mem> weights_buf;
scoped_array<cl_mem> bias_buf;

DTYPE *weights;
DTYPE *biases;
DTYPE *image;
DTYPE *data_init;

scoped_array<int> layers_per_device;
scoped_array<scoped_array<int>> assigned_layers;

// Threads that are handling the devices
scoped_array<pthread_t> device_threads;

unsigned layer_config_original[LAYER_NUM][NUM_CONFIG_ITEM];

void loadImageToBuffer(int num);
int  prepare();
void printCurrentTime();
void cleanup();
void SplitBufferToArray(char *buffer, char * delim, char ** Output);

void* device_runner (void* args);

int main(int argc, char** argv)
{
	cl_int status;

	unsigned int weight_buf_size;
	unsigned int bias_buf_size;
	unsigned int pic_num = 1;

	if (argc < 3){
		printf("Error: wrong commad format, usage:\n");
		printf("%s <binaryfile> [layers,[...]]\n", argv[0]);
		printf("Example: main.exe conv.aocx 1,2,3,4 5,6,7,8\n");
		return EXIT_FAILURE;
	}


	printf("***************************************************\n");
	printf("PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs \n");
	printf("***************************************************\n");

	// Connect to the desired platform
	platform_id = findPlatform(vendor_name);
	if(platform_id == NULL) {
		printf("ERROR: Unable to find the desired OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL device
	device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	num_devices = 2;
	printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("  Device %d: %s\n", i, getDeviceName(device[i]).c_str());
		displayDeviceInfo(device[i]);
	}
	context.reset(num_devices);
	program.reset(num_devices);

	if (num_devices != argc - 2) {
		printf ("ERROR: Number of layer segmentations should be\nequal to the number of devices!\n");
		return false;
	}

	// Create the context.
	context[0] = clCreateContext(NULL, 1, &(device[1]), NULL, NULL, &status);
	checkError(status, "Failed to create context");
	
	context[1] = clCreateContext(NULL, 1, &(device[0]), NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create Program Objects
	char *kernel_file_name=argv[1];

	// Create the program for all device. All devices execute the same kernel.
	program[0] = createProgramFromFile(context[0], (const char *) kernel_file_name, &(device[1]), 1);
	program[1] = createProgramFromFile(context[1], (const char *) kernel_file_name, &(device[0]), 1);

	// Extracting the layer segmentations	
	assigned_layers.reset(num_devices);
	layers_per_device.reset(num_devices);
	for (int i = 0; i < num_devices; i++) {
		int* layers_as_array = new int[20];
		int num_layers_involved = 0;
		char* layers = argv[i+2];
		char* pch = strtok(layers, ",");
		while (pch != NULL) {
			layers_as_array[num_layers_involved] = atoi(pch);
			num_layers_involved++;
			pch = strtok(NULL, ",");
		}
		assigned_layers[i].reset(num_layers_involved);
		layers_per_device[i] = num_layers_involved;

		for (int j = 0; j < num_layers_involved; j++) {
			assigned_layers[i][j] = layers_as_array[j];
		}
	}	

	device_threads.reset(num_devices);

	// Prepare compute data
	status = prepare();
	
	printf ("After prepare!\n");
	if(status == 1) {
		printf("Allocate memory for data and weights failed !!!\n");
		return false;
	}

	// Printing the input information
	printf("[INFO] Number of involved devices are %d\n", num_devices);
	for (int i = 0; i < num_devices; i++) {
		printf("[INFO] layers involved for device %d is: ", i);
		for (int j = 0; j < layers_per_device[i]; j++) {
			printf("%d ", assigned_layers[i][j]);
		}
		printf ("\n");
	}

	
	// create per device object
	que_memRdData.reset(num_devices);
	que_memRdWeight.reset(num_devices);
	que_controller.reset(num_devices);
	que_memWrite.reset(num_devices);
	knl_memRdData.reset(num_devices);
	knl_memRdWeight.reset(num_devices);
	knl_controller.reset(num_devices);
	knl_memWrite.reset(num_devices);
	knl_ser.reset(num_devices);
	knl_deser.reset(num_devices);

	config_buf.reset(num_devices);
	bottom0_buf.reset(num_devices);
	bottom1_buf.reset(num_devices);
	weights_buf.reset(num_devices);
	bias_buf.reset(num_devices);

	// Command queue	
	for (int i = 0; i < num_devices; i++) {
		printf ("[INFO] Creating the command queues for the " ANSI_COLOR_RED "Device %d " ANSI_COLOR_RESET "\n", i);
		que_memRdData[i] = clCreateCommandQueue(context[i], device[(!(i)&1)], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue for memReadData");
		que_memRdWeight[i] = clCreateCommandQueue(context[i], device[(!(i)&1)], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue for memRdWeight");
		que_controller[i] = clCreateCommandQueue(context[i], device[(!(i)&1)], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue for controller");
		que_memWrite[i] = clCreateCommandQueue(context[i], device[(!(i)&1)], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue for memWrite");

		// Kernel
		printf ("[INFO] Creating kernels for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		knl_memRdData[i] = clCreateKernel(program[i], knl_name_memRdData, &status);
		checkError(status, "Failed to create memRdData kernel");

		knl_memRdWeight[i] = clCreateKernel(program[i], knl_name_memRdWeight, &status);
		checkError(status, "Failed to create memRdWeight kernel");

		knl_controller[i] = clCreateKernel(program[i], knl_name_controller, &status);
		checkError(status, "Failed to create controller kernel");

		knl_memWrite[i] = clCreateKernel(program[i], knl_name_memWrite, &status);
		checkError(status, "Failed to create memWrite kernel");

		knl_ser[i] = clCreateKernel(program[i], knl_name_ser, &status);
		checkError(status, "Failed to create ser kernel");

		knl_deser[i] = clCreateKernel(program[i], knl_name_deser, &status);
		checkError(status, "Failed to create deser kernel");

		// Here we should calculate how many items we will have in the whole weight buffer.
		// For each layer, we have the total number of output cannels.
		// Each output channels, is a brick which is of size
		// => num_input_channels * weight_h * W_VEC
		// as you can see, the weight_w is replaced by the W_VEC, because we are using the
			// winograd transformation.
		weight_buf_size = 0;
		for (int layer = 0; layer < LAYER_NUM; layer++) {
			weight_buf_size += (layer_config[layer][weight_m]*layer_config[layer][weight_t]*layer_config[layer][weight_n]*layer_config[layer][weight_h]*W_VEC);
		}
		printf ("[INFO] Creating the weight buffer for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		weights_buf[i] = clCreateBuffer(context[i], CL_MEM_READ_ONLY, weight_buf_size * sizeof(DTYPE), NULL, &status);
		checkError(status, "Failed to create buffer for weights in layer");

		// The total number of biases, is equal to the total number of layers,
		// multiplied by the number of output channels of the each layer
		bias_buf_size = 0;
		for (int layer = 0; layer < LAYER_NUM; layer++) {
			bias_buf_size += layer_config[layer][weight_m];
		}
		printf ("[INFO] Creating the bias buffer for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		bias_buf[i] = clCreateBuffer(context[i], CL_MEM_READ_ONLY, bias_buf_size * sizeof(DTYPE), NULL, &status);
		checkError(status, "Failed to create buffer for bias in layer");

		// Initializing all weights buffers, blocking write is used
		//

		printf ("[INFO] weight_buf_size=%d, WEIGHTS_FILE_SIZE=%d\n", weight_buf_size, WEIGHTS_FILE_SIZE);

		printf ("[INFO] Enqueueing the weight buffer to weight queue for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		status = clEnqueueWriteBuffer(que_memRdWeight[i], weights_buf[i], CL_TRUE, 0, weight_buf_size * sizeof(DTYPE), weights, 0, NULL, NULL);
		checkError(status, "Failed to transfer weight");

		printf ("[INFO] biases_buf_size=%d, BIASES_FILE_SIZE=%d\n", bias_buf_size, BIASES_FILE_SIZE);	

		printf ("[INFO] Enqueueing the bias buffer to bias queue for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		status = clEnqueueWriteBuffer(que_memRdWeight[i], bias_buf[i], CL_TRUE, 0, bias_buf_size * sizeof(DTYPE), biases, 0, NULL, NULL);
		checkError(status, "Failed to transfer bias");
		
		// First buffer
		printf ("[INFO] Creating the bottom0 buffer for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		bottom0_buf[i] = clCreateBuffer(context[i], CL_MEM_READ_WRITE, IN_BUF_SIZE * sizeof(DTYPE), NULL, &status);
		checkError(status, "Failed to create buffer for data in layer");

		// Second buffer
		printf ("[INFO] Creating the bottom1 buffer for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", i);
		bottom1_buf[i] = clCreateBuffer(context[i], CL_MEM_READ_WRITE, OUT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
		checkError(status, "Failed to create buffer for output");

		//printf ("[INFO] Enqueueing the bottom0 buffer\n");
		//status = clEnqueueWriteBuffer(que_memRdData, bottom0_buf, CL_TRUE, 0, IN_BUF_SIZE * sizeof(DTYPE), NULL, 0, NULL, NULL);
		//checkError(status, "Failed to transfer bottom0");

		//printf ("[INFO] Enqueueing the bottom1 buffer\n");
		//status = clEnqueueWriteBuffer(que_memRdData, bottom1_buf, CL_TRUE, 0, OUT_BUF_SIZE * sizeof(DTYPE), NULL, 0, NULL, NULL);
		//checkError(status, "Failed to transfer bottom1");

		printf ("[INFO] Creating the config buffer with size %d\n", sizeof(fpga_configuration) * LAYER_NUM);
		config_buf[i] = clCreateBuffer(context[i], CL_MEM_READ_WRITE, sizeof(fpga_configuration) * (layers_per_device[i]+1), NULL, &status);
		checkError(status, "Failed to create buffer for config");

		fpga_configuration fpga_config[layers_per_device[i]+1];
		//printf ("[INFO] Layers_per_device+1=%d\n", layers_per_device[i]+1);
		//for (int j = 0; j < layers_per_device[i]+1; j++) {
		//	printf("[INFO] %d,%d\n",j, config[j].layer_type);
		//}

		printf ("[INFO] Setting the configurations per layer (for %d layers) for the " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "\n", layers_per_device[i]+1, i);
		for (int layer = 0; layer < layers_per_device[i]+1; layer++) {

			//printf ("[INFO] Working on the layer #%d\n", layer);
			fpga_config[layer].layer_type = layer_config[assigned_layers[i][0]+layer-1][layer_type];
			
			//printf ("[INFO] Layer type is %d\n", layer_config[assigned_layers[i][0]+layer][layer_type]);
			fpga_config[layer].data_w = layer_config[assigned_layers[i][0]+layer-1][data_w];
			fpga_config[layer].data_h = layer_config[assigned_layers[i][0]+layer-1][data_h];
			fpga_config[layer].data_t = layer_config[assigned_layers[i][0]+layer-1][data_t];
			fpga_config[layer].weight_w = layer_config[assigned_layers[i][0]+layer-1][weight_w];
			fpga_config[layer].weight_h = layer_config[assigned_layers[i][0]+layer-1][weight_h];
			fpga_config[layer].weight_n = layer_config[assigned_layers[i][0]+layer-1][weight_n];
			fpga_config[layer].weight_t = layer_config[assigned_layers[i][0]+layer-1][weight_t];
			fpga_config[layer].weight_m = layer_config[assigned_layers[i][0]+layer-1][weight_m];
			fpga_config[layer].memrd_src = layer_config[assigned_layers[i][0]+layer-1][memrd_src];
			fpga_config[layer].conv_x = layer_config[assigned_layers[i][0]+layer-1][conv_x];
			fpga_config[layer].conv_y = layer_config[assigned_layers[i][0]+layer-1][conv_y];
			fpga_config[layer].conv_z = layer_config[assigned_layers[i][0]+layer-1][conv_z];
			fpga_config[layer].conv_t = layer_config[assigned_layers[i][0]+layer-1][conv_t];
			fpga_config[layer].conv_stride = layer_config[assigned_layers[i][0]+layer-1][conv_stride];
			fpga_config[layer].conv_padding = layer_config[assigned_layers[i][0]+layer-1][conv_padding];
			fpga_config[layer].conv_split = layer_config[assigned_layers[i][0]+layer-1][conv_split];
			fpga_config[layer].conv_relu = layer_config[assigned_layers[i][0]+layer-1][conv_relu];
			fpga_config[layer].pool_on = layer_config[assigned_layers[i][0]+layer-1][pool_on];
			fpga_config[layer].pool_x = layer_config[assigned_layers[i][0]+layer-1][pool_x];
			fpga_config[layer].pool_y = layer_config[assigned_layers[i][0]+layer-1][pool_y];
			fpga_config[layer].pool_z = layer_config[assigned_layers[i][0]+layer-1][pool_z];
			fpga_config[layer].pool_t = layer_config[assigned_layers[i][0]+layer-1][pool_t];
			fpga_config[layer].pool_size_xy = layer_config[assigned_layers[i][0]+layer-1][pool_size_xy];
			fpga_config[layer].pool_size_t = layer_config[assigned_layers[i][0]+layer-1][pool_size_t];
			fpga_config[layer].pool_stride_xy = layer_config[assigned_layers[i][0]+layer-1][pool_size_xy];
			fpga_config[layer].pool_stride_t = layer_config[assigned_layers[i][0]+layer-1][pool_size_t];
			fpga_config[layer].lrn_on = layer_config[assigned_layers[i][0]+layer-1][lrn_on];
			fpga_config[layer].memwr_dst = layer_config[assigned_layers[i][0]+layer-1][memwr_dst];

			//printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "layer_type: %d, data_w: %d, data_h: %d, weight_w: %d, weight_h: %d, weight_n: %d, weight_m: %d, memrd_src: %d, conv_x: %d, conv_y: %d, conv_z: %d, conv_stride: %d, conv_padding: %d, conv_split: %d, conv_relu: %d, pool_on: %d, pool_x: %d, pool_y: %d, pool_z: %d, pool_size: %d, conv_stride: %d, lrn_on: %d, memwr_dst: %d\n", i, layer_config[assigned_layers[i][0]+layer][layer_type], layer_config[assigned_layers[i][0]+layer][data_w], layer_config[assigned_layers[i][0]+layer][data_h], layer_config[assigned_layers[i][0]+layer][weight_w], layer_config[assigned_layers[i][0]+layer][weight_h], layer_config[assigned_layers[i][0]+layer][weight_n], layer_config[assigned_layers[i][0]+layer][weight_m], layer_config[assigned_layers[i][0]+layer][memrd_src], layer_config[assigned_layers[i][0]+layer][conv_x], layer_config[assigned_layers[i][0]+layer][conv_y], layer_config[assigned_layers[i][0]+layer][conv_z], layer_config[assigned_layers[i][0]+layer][conv_stride], layer_config[assigned_layers[i][0]+layer][conv_padding], layer_config[assigned_layers[i][0]+layer][conv_split], layer_config[assigned_layers[i][0]+layer][conv_relu], layer_config[assigned_layers[i][0]+layer][pool_on], layer_config[assigned_layers[i][0]+layer][pool_x], layer_config[assigned_layers[i][0]+layer][pool_y], layer_config[assigned_layers[i][0]+layer][pool_z], layer_config[assigned_layers[i][0]+layer][pool_size], layer_config[assigned_layers[i][0]+layer][pool_stride], layer_config[assigned_layers[i][0]+layer][lrn_on], layer_config[assigned_layers[i][0]+layer][memwr_dst]);
		
			// int w_vec = W_VEC;
			// fpga_config[layer].num_bricks = (layer_config[assigned_layers[i][0]+layer][data_h]+2*layer_config[assigned_layers[i][0]+layer][conv_padding]-layer_config[assigned_layers[i][0]+layer][weight_h]+1)*((layer_config[assigned_layers[i][0]+layer][data_w]+2*layer_config[assigned_layers[i][0]+layer][conv_padding]-layer_config[assigned_layers[i][0]+layer][weight_w])/(W_VEC-layer_config[assigned_layers[i][0]+layer][weight_w]+1) + 1);

		
			int w_vec = W_VEC;
			int num_bricks_w = (fpga_config[layer].data_w+2*fpga_config[layer].conv_padding-fpga_config[layer].weight_w)/(W_VEC-fpga_config[layer].weight_w+1) + 1;
			int num_bricks_h = fpga_config[layer].data_h+2*fpga_config[layer].conv_padding-fpga_config[layer].weight_h+1;
			int num_bricks_t = fpga_config[layer].data_t+2*fpga_config[layer].conv_padding-fpga_config[layer].weight_t+1;
			fpga_config[layer].num_bricks = num_bricks_w * num_bricks_h * num_bricks_t;
			printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "conv_x=%d, conv_y=%d, conv_z=%d\n", i, fpga_config[layer].conv_x, fpga_config[layer].conv_y, fpga_config[layer].conv_z);

			printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "w_vec: %d\n", i, w_vec);
			printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "data_w: %d\n", i, fpga_config[layer].data_w);
			printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "data_h: %d\n", i, fpga_config[layer].data_h);
			printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "conv_padding: %d\n", i, fpga_config[layer].conv_padding);
			printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "weight_w: %d\n", i, fpga_config[layer].weight_w);	
			//printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "Some #1: %d\n", i, layer_config[layer][data_h]+2*layer_config[assigned_layers[i][0]+layer][conv_padding]-layer_config[assigned_layers[i][0]+layer][weight_h]+1);
			//printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "first part: %d\n", i, (layer_config[assigned_layers[i][0]+layer][data_w]+2*layer_config[assigned_layers[i][0]+layer][conv_padding]-layer_config[assigned_layers[i][0]+layer][weight_w]));
			//printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "second part: %d\n", i, (w_vec-layer_config[assigned_layers[i][0]+layer][weight_w]+1));
			//printf ("[INFO] " ANSI_COLOR_RED "DEVICE %d " ANSI_COLOR_RESET "Some #2: %d\n", i, ((layer_config[assigned_layers[i][0]+layer][data_w]+2*layer_config[assigned_layers[i][0]+layer][conv_padding]-layer_config[assigned_layers[i][0]+layer][weight_w])/(w_vec-layer_config[assigned_layers[i][0]+layer][weight_w]+1)) + 1);
			// printf ("[INFO] Some #1: %d, Some #2: %d, data_w: %d, conv_padding: %d, w_vec: %d, weight_w: %d\n", layer_config[layer][data_h]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_h]+1, ceil((layer_config[layer][data_w]+2*layer_config[layer][conv_padding]-W_VEC)/(W_VEC-layer_config[layer][weight_w]+1)), layer_config[layer][data_w], layer_config[layer][conv_padding], w_vec, layer_config[layer][weight_w]);
		}
	
		printf ("[INFO] Enqueueing the config buffer to controller queue for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "\n", i);
		status = clEnqueueWriteBuffer(que_controller[i], config_buf[i], CL_TRUE, 0, sizeof(fpga_configuration) * (layers_per_device[i]+1), fpga_config, 0, NULL, NULL);
		checkError(status, "Failed to transfer config");

	}

	
	device_threads.reset(num_devices);
	for (int i = 0; i < num_devices; i++) {
		device_runner_arg* device_arg = new device_runner_arg;
		device_arg->device = i;

		printf ("[INFO] Dispatching thread #%d\n", i);	
		pthread_create(&(device_threads[i]), NULL, device_runner, (void *) device_arg);
	}

	for (int i = 0; i < num_devices; i++) {
		printf ("[INFO] Waiting for device %d\n", i);
		pthread_join((device_threads[i]), NULL);
		printf ("[INFO] device %d has joined!\n", i);
	}
	

	/*
	device_runner_arg* device_arg = new device_runner_arg;
	device_arg->device = 0;
	pthread_create(&(device_threads[0]), NULL, device_runner, (void *) device_arg);
	
	printf ("[INFO] Dispatching thread #1\n");

	pthread_join((device_threads[0]), NULL);
	*/


	//Recorde the end time
	printf("\nPipeCNN exited !!!\n\n");

	// Release resource
	cleanup();

	return EXIT_SUCCESS;
}

void loadImageToBuffer(int num)
{
	cl_int status;
	ifstream bin_file_r;

	unsigned file_size;
	// load image from binary files
	bin_file_r.open(input_file_path, ios::in | ios::binary);

    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);

    	bin_file_r.read((char *)image, sizeof(DTYPE)*IMAGE_FILE_SIZE);
    	printf("\n%d bytes image data read from binary files\n", file_size);
		if(IMAGE_FILE_SIZE!=(file_size/(sizeof(DTYPE))))
			printf("Warning: image file size does not match user configuration !!!\n");
    	bin_file_r.close();
    }
    else
		printf("Image file does not exits !!!\n");

	// Vectorize the input image by a factor of VEC_SIZE
	for(unsigned n = 0; n<layer_config[0][data_n]/VEC_SIZE; n++){
		for(unsigned i = 0; i<layer_config[0][data_h]; i++){
			for(unsigned j = 0; j<layer_config[0][data_w]; j++){
				for(unsigned k = 0; k<VEC_SIZE; k++){
					if((n*VEC_SIZE+k)<layer_config_original[0][data_n]){ //  when layer_config[0][data_n] > layer_config_original[0][data_n], only copy valid pixels
						data_init[n*VEC_SIZE*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w]*VEC_SIZE + j*VEC_SIZE + k]
							= (DTYPE) image[(n*VEC_SIZE+k)*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w] + j];
					}
				}
			}
		}
	}

	// Load image data into buffers
	status = clEnqueueWriteBuffer(que_memRdData[0], bottom0_buf[0], CL_TRUE, 0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
	checkError(status, "Failed to transfer input image");

#ifdef VERBOSE_OUTPUT
	DTYPE* temp_output = new DTYPE[layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]];
	status = clEnqueueReadBuffer(que_memRdData, bottom0_buf, CL_TRUE, 0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), (void *) temp_output, 0, NULL, NULL);
	checkError (status, "Failed to read back the data");

	char fileName[20] = {'\0'};
	sprintf (fileName, "Input.txt");

	FILE* fp;
	fp = fopen(fileName, "w");

	for (int i = 0; i < layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]; i++) {
		fprintf (fp, "%f\n", (float) temp_output[i]);
#endif
}

// Read all input data and golden ref data
int prepare()
{

	// Load Image data, CNN net weights and golden_results
 	ifstream bin_file_r;
	unsigned file_size;

	unsigned char  conv_win_size_dim1, conv_win_size_dim2;

	unsigned padding_offset[LAYER_NUM];

	// Parameter initialization and safty check
	for(unsigned ll=0; ll<LAYER_NUM; ll++){

		// First, backup the original layer configurations
		for(unsigned ii=0; ii<NUM_CONFIG_ITEM; ii++){
			layer_config_original[ll][ii]=layer_config[ll][ii];
		}

		// Second, perform padding on dim4, when it is not divisible by LANE_NUM
		if(layer_config[ll][weight_m]%LANE_NUM != 0){
			printf("\nWarnning: layer-%d requires padding zero-value feature maps for give param LANE_NUM=%d\n", ll+1, LANE_NUM);
			layer_config[ll][weight_m] = ceil((float)layer_config[ll][weight_m]/LANE_NUM)*LANE_NUM;
 	      		layer_config[ll][bias_size] = layer_config[ll][weight_m];
			printf("      original num of feature maps is %d, new value is %d\n", layer_config_original[ll][weight_m], layer_config[ll][weight_m]);
			
			// padding of weight on dim4 is needed
			padding_offset[ll] = layer_config[ll][weight_m] - layer_config_original[ll][weight_m];

			// check if evenly padding on two sides is possible
			if(((layer_config[ll][weight_m]/LANE_NUM)%2!=0) & (layer_config[ll][conv_split]==1)){
				printf("Error: could not perform padding for split mode, weight_m/LANE_NUM must be divisible by 2 !!!\n\n");
				return 1;
			}
			else{ // padding zeros evenly on two sides of dim4
				padding_offset[ll] = padding_offset[ll]/2;
				printf("      padding_offset=%d (layer=%d)\n\n", padding_offset[ll], ll+1);
			}

		}
		else{
			padding_offset[ll] = 0;
		}

		// Check parameters
		if(ll==0){ // check parameters for layer-1
			if(input_config[image_w] != layer_config_original[ll][data_w] ||  input_config[image_h] != layer_config_original[ll][data_h]
				|| input_config[image_n] != layer_config_original[ll][data_n] || input_config[image_t] != layer_config_original[ll][data_t]){
					printf("Error: incorrect layer configuration for layer-%d !!!\n", ll+1);
					//return;
			}

			if((layer_config_original[ll][weight_n]!=input_config[image_n])){
				printf("\nError: incorrect layer configuration for layer-%d !!!\n", ll+1);
				//return 1;
			}

		}
		else{ // other layers
			// Currently weight_n must be divisible by VEC_SIZE (for first layer, padding is performed when weight_n is not divisible by VEC_SIZE)
			if((layer_config[ll][weight_n]%VEC_SIZE)!=0){
				printf("\nError: incorrect setting of parameter VEC_SIZE !!!\n");
				//return 1;
			}
			if((layer_config_original[ll][data_n]!=layer_config_original[ll-1][conv_z])){
				printf("\nError: incorrect setting of convolution input/output size for layer-%d!!!\n", ll+1);
				//return 1;
			}
		}
		if((layer_config_original[ll][conv_x]!=(layer_config_original[ll][data_w]-layer_config_original[ll][weight_w]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
			|| (layer_config_original[ll][conv_y]!=(layer_config_original[ll][data_h]-layer_config_original[ll][weight_h]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
			|| (layer_config_original[ll][conv_t]!=(layer_config_original[ll][data_t]-layer_config_original[ll][weight_t]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
			|| (layer_config_original[ll][conv_z]!=layer_config_original[ll][weight_m])){
				printf("\nError: incorrect setting of convolution output size or filter params for layer-%d!!!\n", ll+1);
				//return 1;
		}
		if(layer_config_original[ll][pool_on] && ((layer_config_original[ll][pool_x]!=(layer_config_original[ll][conv_x]-layer_config_original[ll][pool_size_xy])/layer_config_original[ll][pool_stride_xy]+1)
			|| (layer_config_original[ll][pool_y]!=(layer_config_original[ll][conv_y]-layer_config_original[ll][pool_size_xy])/layer_config_original[ll][pool_stride_xy]+1)
			|| (layer_config_original[ll][pool_t]!=(layer_config_original[ll][conv_t]-layer_config_original[ll][pool_size_t])/layer_config_original[ll][pool_stride_t]+1)
			|| (layer_config_original[ll][pool_z]!=layer_config_original[ll][conv_z]))){
				printf("\nError: incorrect setting of pooling input/output size for layer-%d!!!\n", ll+1);
				//return 1;
		}
		if(layer_config[ll][conv_x]==1){ // when only one group for FC layer
			conv_win_size_dim1  = layer_config[ll][weight_w];
		}
		else{
			conv_win_size_dim1  = layer_config[ll][weight_w]+(CONV_GP_SIZE_X-1)*layer_config[ll][conv_stride];
		}
		conv_win_size_dim2    = layer_config[ll][weight_h];
		// check win_buffer size
		/*
		if(conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/VEC_SIZE > WIN_BUF_SIZE){
			printf("Error: required win_buffer size is %d, configured size is %d, because win_size_dim1=%d and win_size_dim2=%d and weight_n=%d\n", conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/VEC_SIZE, WIN_BUF_SIZE, conv_win_size_dim1, conv_win_size_dim2, layer_config[ll][weight_n]);
			return 1;
		}
		// check weight_buffer size
		if(layer_config[ll][weight_w]*layer_config[ll][weight_h]*layer_config[ll][weight_n]/VEC_SIZE > WEIGHT_BUF_SIZE){
			printf("Error: required weight_buffer size is %d, configured size is %d \n", layer_config[ll][weight_w]*layer_config[ll][weight_h]*layer_config[ll][weight_n]/VEC_SIZE, WEIGHT_BUF_SIZE);
			return 1;
		}
		*/
	}

	// image and weight files
	weights      = (DTYPE *)alignedMalloc(sizeof(DTYPE)*WEIGHTS_FILE_SIZE, DMA_ALIGNMENT);
	image        = (DTYPE *)alignedMalloc(sizeof(DTYPE)*IMAGE_FILE_SIZE, DMA_ALIGNMENT);
	biases	     = (DTYPE *)alignedMalloc(sizeof(DTYPE)*BIASES_FILE_SIZE, DMA_ALIGNMENT);
	// input data buffers
	// padding the input RGB image with extrals, so that data_n/weight_n is divisible by VEC_SIZE
	layer_config[0][weight_n] = ceil((float)layer_config[0][weight_n]/VEC_SIZE)*VEC_SIZE;
	printf ("[INFO] weight_n is changed to %d\n", layer_config[0][weight_n]);
	layer_config[0][data_n] = layer_config[0][weight_n];

	data_init   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]*layer_config[0][data_t], DMA_ALIGNMENT);
	memset(data_init, 0, sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]*layer_config[0][data_t]);// fill non-RGB dims with 0

	if(weights == NULL || image == NULL || data_init == NULL || biases == NULL)
	{
		printf("Not enough memory !!!");
		alignedFree(weights);
		alignedFree(biases);
		alignedFree(image);
		alignedFree(data_init);

		return 1;
	}

	// Weights
	bin_file_r.open(weight_file_path, ios::in | ios::binary);

	if(bin_file_r.is_open())
	{
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);

		bin_file_r.read((char *)weights, sizeof(DTYPE)*WEIGHTS_FILE_SIZE);
		printf("\n%d total weights read \n", file_size/((int)sizeof(DTYPE)));
		if(WEIGHTS_FILE_SIZE!=(file_size/(sizeof(DTYPE))))
			printf("Warning: weight file size does not match user configuration !!!\n");
		bin_file_r.close();
	}
	else
		printf("Weights file does not exits !!!\n");
	int status = EXIT_SUCCESS;//reorderWeights(padding_offset);
	if(status != EXIT_SUCCESS) {
		printf("Error in reorderWeights : %d\n", status);
		return status;
	}
	return EXIT_SUCCESS;
}

// Release all memory resources here
void cleanup()
{

	// Release the opencl runtime resource allocated
	for(unsigned i = 0; i < 1; ++i) {

		printf ("[INFO] Killing all kernels!\n");

		// Killing the kernels
		if(knl_memRdData[0]) {
			clReleaseKernel(knl_memRdData[0]);
		}
		if(knl_memRdData[1]) {
			clReleaseKernel(knl_memRdData[1]);
		}
		if(knl_memRdWeight[0]) {
			clReleaseKernel(knl_memRdWeight[0]);
		}
		if(knl_memRdWeight[1]) {
			clReleaseKernel(knl_memRdWeight[1]);
		}
		if(knl_controller[0]) {
			clReleaseKernel(knl_controller[0]);
		}
		if(knl_controller[1]) {
			clReleaseKernel(knl_controller[1]);
		}
		if(knl_memWrite[0]) {
			clReleaseKernel(knl_memWrite[0]);
		}
		if(knl_memWrite[1]) {
			clReleaseKernel(knl_memWrite[1]);
		}

		printf ("[INFO] Killing all command queues!\n");

		// Killing all the queues
		if(que_memRdData[0]) {
			clReleaseCommandQueue(que_memRdData[0]);
		}
		if(que_memRdData[1]) {
			clReleaseCommandQueue(que_memRdData[1]);
		}
		if(que_memRdWeight[0]) {
			clReleaseCommandQueue(que_memRdWeight[0]);
		}
		if(que_memRdWeight[1]) {
			clReleaseCommandQueue(que_memRdWeight[1]);
		}
		if(que_controller[0]) {
			clReleaseCommandQueue(que_controller[0]);
		}
		if(que_controller[1]) {
			clReleaseCommandQueue(que_controller[1]);
		}
		if(que_memWrite[0]) {
			clReleaseCommandQueue(que_memWrite[0]);
		}
		if(que_memWrite[1]) {
			clReleaseCommandQueue(que_memWrite[1]);
		}

		printf ("[INFO] Releasing all memory objects!\n");

		// Killing all the buffers
		if(config_buf[0]) {
			clReleaseMemObject(config_buf[0]);
		}
		if(config_buf[1]) {
			clReleaseMemObject(config_buf[1]);
		}
		if(bottom0_buf[0]) {
			clReleaseMemObject(bottom0_buf[0]);
		}
		if(bottom0_buf[1]) {
			clReleaseMemObject(bottom0_buf[1]);
		} 
		if(bottom1_buf[0]) {
			clReleaseMemObject(bottom1_buf[0]);
		}
		if(bottom1_buf[1]) {
			clReleaseMemObject(bottom1_buf[1]);
		}
		if(weights_buf[0]) {
			clReleaseMemObject(weights_buf[0]);
		}
		if(weights_buf[1]) {
			clReleaseMemObject(weights_buf[1]);
		}
		if(bias_buf[0]) {
			clReleaseMemObject(bias_buf[0]);
		}
		if(bias_buf[1]) {
			clReleaseMemObject(bias_buf[1]);
		}
	}

	printf ("[INFO] Releasing all programs!\n");

	if(program[0]) {
		clReleaseProgram(program[0]);
	}
	if(program[1]) {
		clReleaseProgram(program[1]);
	}
	if(context[0]) {
		clReleaseContext(context[0]);
	}
	if(context[1]) {
		clReleaseContext(context[1]);
	}

	printf ("[INFO] Deallocating host buffers!\n");

	alignedFree(weights);
	alignedFree(image);
	alignedFree(biases);

}

void printCurrentTime() {

	char fmt[64];
	char buf[64];
	
	struct timeval tv;
	struct tm* tm;

	gettimeofday(&tv, NULL);
	tm = localtime (&tv.tv_sec);
	strftime (fmt, sizeof (fmt), "%H:%M:%S:%%6u", tm);
	snprintf (buf, sizeof (buf), fmt, tv.tv_usec);
	printf ("[INFO] Reading at %s\n", buf);

}

void SplitBufferToArray(char *buffer, char * delim, char ** Output) {

    int partcount = 0;
    Output[partcount++] = buffer;

    char* ptr = buffer;
    while (ptr != 0) { //check if the string is over
        ptr = strstr(ptr, delim);
        if (ptr != NULL) {
            *ptr = 0;
            Output[partcount++] = ptr + strlen(delim);
            ptr = ptr + strlen(delim);
        }

    }
    Output[partcount++] = NULL;
}

void* device_runner (void* args) {

	cl_int status;
	unsigned int pic_num = 1;

	device_runner_arg *device_arg = (device_runner_arg *) args;
	int i = device_arg->device;
	char i_ch = i;

	// Execute the kernel
	cl_event deser_event;
	cl_event memRdData_event;
	cl_event memRdWeight_event;
	cl_event memWrite_event;
	cl_event ser_event;
	cl_event controller_event;

	// Recorde the excution time of each operation
	cl_ulong memRdData_time;
	cl_ulong memRdWeight_time;
	cl_ulong memWrite_time;
	cl_ulong controller_time;


	for (int iter = 0; iter < 4; iter++) {

		if (i == 0)	
			loadImageToBuffer(pic_num);
		unsigned argi = 0;
		char layer_num = layers_per_device[i];

		// Setting the arguments for the controller
		argi = 0;

		printf ("[INFO] Setting kernel arguments for the controller for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "! i_ch=%d, layer_num=%d\n", i, i_ch, layer_num);
		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &i_ch);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &layer_num);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		// Only the first device avoids deserialization of the data
		char deser_data;
		if (i == 0) deser_data = 0;
		else deser_data = 1;

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &deser_data);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		char ser_data;
		if (i == num_devices-1) ser_data = 0;
		else ser_data = 1;

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &ser_data);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &precision_config[0][frac_w]);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &precision_config[0][frac_din]);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_char), &precision_config[0][frac_dout]);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller[i], argi++, sizeof(cl_mem), &(config_buf[i]));
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);


		// Setting the arguments for the deser module
		argi = 0;
		
		printf("[INFO] Setting kernel arguments for the deser for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "! i_ch=%d, deser_data=%d\n", i, i_ch, deser_data);
		status = clSetKernelArg(knl_deser[i], argi++, sizeof(cl_char), &i_ch);
		checkError(status, "Failed to set argument %d of kernel deser", argi-1);

		status = clSetKernelArg(knl_deser[i], argi++, sizeof(cl_char), &deser_data);
		checkError(status, "Failed to set argument %d of kernel deser", argi-1);

		cl_mem *bottom;
		if (assigned_layers[i][0] % 2 == 0) bottom = &(bottom0_buf[i]);
		else bottom = &(bottom1_buf[i]);

		status = clSetKernelArg(knl_deser[i], argi++, sizeof(cl_mem), bottom);
		checkError(status, "Failed to set argument %d of kernel deser", argi-1);

		// Setting the arguments for the memory read data module
		argi = 0;
		char config_size = layer_num;
		char start_buffer;
		if (assigned_layers[i][0] % 2 == 0) start_buffer = 0x00;
		else start_buffer = 0x01;

		printf ("[INFO] Setting kernel arguments for the memRdData for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "! i_ch=%d, config_size=%d, start_buffer=%d\n", i, i_ch, config_size, start_buffer);
		status = clSetKernelArg(knl_memRdData[i], argi++, sizeof(cl_char), &i_ch);
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		status = clSetKernelArg(knl_memRdData[i], argi++, sizeof(cl_char), &config_size);
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);
		
		status = clSetKernelArg(knl_memRdData[i], argi++, sizeof(cl_char), &start_buffer);
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		status = clSetKernelArg(knl_memRdData[i], argi++, sizeof(cl_mem), &(bottom0_buf[i]));
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		status = clSetKernelArg(knl_memRdData[i], argi++, sizeof(cl_mem), &(bottom1_buf[i]));
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		argi = 0;

		printf ("[INFO] Setting kernel arguments for the memRdWeight for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "! config_size=%d\n", i, config_size);
		status = clSetKernelArg(knl_memRdWeight[i], argi++, sizeof(cl_char), &i_ch);
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		status = clSetKernelArg(knl_memRdWeight[i], argi++, sizeof(cl_char), &config_size);
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		status = clSetKernelArg(knl_memRdWeight[i], argi++, sizeof(cl_mem), &(weights_buf[i]));
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		status = clSetKernelArg(knl_memRdWeight[i], argi++, sizeof(cl_mem), &(bias_buf[i]));
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		argi = 0;
		if (assigned_layers[i][0] % 2 == 0) start_buffer = 0x01;
		else start_buffer = 0x00;
		
		printf ("[INFO] Setting kernel arguments for the memWrite for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "! i_ch=%d, config_size=%d, start_buffer=%d\n", i, i_ch, config_size, start_buffer);
		status = clSetKernelArg(knl_memWrite[i], argi++, sizeof(cl_char), &i_ch);
		checkError(status, "Failed to set argument %d of kernel memory write", argi-1);

		status = clSetKernelArg(knl_memWrite[i], argi++, sizeof(cl_char), &config_size);
		checkError(status, "Failed to set argument %d of kernel memory write", argi-1);

		status = clSetKernelArg(knl_memWrite[i], argi++, sizeof(cl_char), &start_buffer);
		checkError(status, "Failed to set argument %d of kernel memory write", argi-1);

		status = clSetKernelArg(knl_memWrite[i], argi++, sizeof(cl_mem), &(bottom0_buf[i]));
		checkError(status, "Failed to set argument %d of kernel memory write", argi-1);

		status = clSetKernelArg(knl_memWrite[i], argi++, sizeof(cl_mem), &(bottom1_buf[i]));
		checkError(status, "Failed to set argument %d of kernel memory write", argi-1);


		argi = 0;

		cl_mem* top;
		if (assigned_layers[i][layers_per_device[i]-1] % 2 == 0) top = &(bottom1_buf[i]);
		else top = &(bottom0_buf[i]);

		printf ("[INFO] Setting kernel arguments for the ser for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "! i_ch=%d, ser_data=%d\n", i, i_ch, ser_data);
		status = clSetKernelArg(knl_ser[i], argi++, sizeof(cl_char), &i_ch);
		checkError(status, "Failed to set argument %d of kernel ser", argi-1);

		status = clSetKernelArg(knl_ser[i], argi++, sizeof(cl_char), &ser_data);
		checkError(status, "Failed to set argument %d of kernel ser", argi-1);
	
		status = clSetKernelArg(knl_ser[i], argi++, sizeof(cl_mem), top);
		checkError(status, "Failed to set argument %d of kernel ser", argi-1);

		if (i == 0)
			printCurrentTime();

		// Enqueueing kernels
		printf ("[INFO] Enqueuing tasks [controller,deser[if],memRdData,memRdWeight,memWrite,ser[if]] " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		printf ("[INFO] Enqueuing tasks controller " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		status = clEnqueueTask(que_controller[i], knl_controller[i], 0, NULL, &controller_event);
		checkError(status, "Failed to launch kernel controller");

		if (i != 0) {
			printf ("[INFO] Enqueuing tasks deser " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
			status = clEnqueueTask(que_memRdData[i], knl_deser[i], 0, NULL, &deser_event);
			checkError(status, "Failed to lauch kernel deserializer");
		}
	
		printf ("[INFO] Enqueuing tasks memRdData " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		status = clEnqueueTask(que_memRdData[i], knl_memRdData[i], 0, NULL, &memRdData_event);
		checkError(status, "Failed to launch kernel memory read data");

		printf ("[INFO] Enqueuing tasks memRdWeight " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		status = clEnqueueTask(que_memRdWeight[i], knl_memRdWeight[i], 0, NULL, &memRdWeight_event);
		checkError(status, "Failed to launch kernel memory read weight");

		printf ("[INFO] Enqueuing tasks memWrite " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		status = clEnqueueTask(que_memWrite[i], knl_memWrite[i], 0, NULL, &memWrite_event);
		checkError(status, "Failed to launch kernel write");

		if (i != num_devices -1) {
			printf ("[INFO] Enqueuing tasks ser " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
			status = clEnqueueTask(que_memWrite[i], knl_ser[i], 1, &memWrite_event, &ser_event);
			checkError(status, "Failed to launch kernel serializer");
		}

		// Waiting for the events
		status = clWaitForEvents(1, &controller_event);
		checkError(status, "Failed to wait for the controller kernel\n");
		printf ("[INFO] Done with the controller for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);

		if (i != 0) {
			status = clWaitForEvents(1, &deser_event);
			checkError(status, "Failed to wait for the deserializer kernel\n");
			printf ("[INFO] Done with the deser for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		}

		status = clWaitForEvents(1, &memRdData_event);
		checkError(status, "Failed to wait for the memRdData kernel\n");
		printf ("[INFO] Done with the memReadData for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		
		status = clWaitForEvents(1, &memRdWeight_event);
		checkError(status, "Failed to wait for the memRdWeight kernel\n");
		printf ("[INFO] Done with the memReadWeight for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);

		status = clWaitForEvents(1, &memWrite_event);
		checkError(status, "Failed to wait for the memWrite kernel\n");
		printf ("[INFO] Done with memWrite for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);

		if (i != num_devices-1) {
			printf ("Waiting for ser\n");
			status = clWaitForEvents(1, &ser_event);
			checkError(status, "Failed to wait for the ser kernel\n");
			printf ("[INFO] Done with ser for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		}

		//printCurrentTime();

		printf ("[INFO] Calculating kernel runtime for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
		memRdData_time = getKernelStartEndTime(memRdData_event, "memRd");
		memRdWeight_time = getKernelStartEndTime(memRdWeight_event, "conv");
		memWrite_time = getKernelStartEndTime(memWrite_event, "memWr");
		controller_time = getKernelStartEndTime(controller_event, "lrn");

		// Must release event object to avoid performance degeneration !!!

		printf ("[INFO] Releasing events for the " ANSI_COLOR_RED "DEVICE %d" ANSI_COLOR_RESET "!\n", i);
	
		if (i != 0) {
			status = clReleaseEvent(deser_event);
			checkError(status, "Failed to release deser data event object");
		}
		status = clReleaseEvent(memRdData_event);
		checkError(status, "Failed to release mem read data event object");
		status = clReleaseEvent(memRdWeight_event);
		checkError(status, "Failed to release mem read weight event object");
		status = clReleaseEvent(memWrite_event);
		checkError(status, "Failed to release mem write event object");
		if (i != num_devices-1) {
			status = clReleaseEvent(ser_event);
			checkError(status, "Failed to release ser data event object");
		}
		status = clReleaseEvent(controller_event);
		checkError(status, "Failed to release controller event object");

	} // end of iterations

}
