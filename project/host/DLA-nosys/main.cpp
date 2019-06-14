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
#define LAYER_NUM         8
#define CONV_NUM          8
const char *weight_file_path = "./data/data_vgg16/weights.dat";
const char *input_file_path = "./data/data_vgg16/image.dat";
#endif

// Configuration file instructions
enum config_item{
layer_type, // "0" -> conv, "1" -> fc

data_w, data_h, data_n, weight_w, weight_h, weight_n, weight_m, bias_size, //memRd Parameters

memrd_src, //"0"-> data_buf  "1"-> output_buf  "2"->"fc_1_buffer"  "3"->"fc_2_buffer"

conv_x, conv_y, conv_z, conv_stride, conv_padding, conv_split, conv_relu, //Conv Parameters

pool_on, pool_x, pool_y, pool_z, pool_size, pool_stride, // Pooling Parameters

lrn_on,// lrn on/off control

memwr_dst//"0"-> data_buf  "1"-> output_buf  "2"->"fc_1_buffer"  "3"->"fc_2_buffer"
};

enum input_item{
image_w, image_h, image_n, // original image size
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
	int data_w, data_h, weight_w, weight_h, weight_n, weight_m, bias_size;
	int memrd_src;
	int conv_x, conv_y, conv_z, conv_stride, conv_padding, conv_split, conv_relu;
	int pool_on, pool_x, pool_y, pool_z, pool_size, pool_stride;
	int lrn_on;
	int memwr_dst;
	int num_bricks;
} fpga_configuration;

// Define the kernel names used
const char *knl_name_memRdData = "memReadData";
const char *knl_name_memRdWeight = "memReadWeight";
const char *knl_name_controller = "controller";
const char *knl_name_memWrite = "memWrite";

//------------ Global Functions & Variables ------------//
cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
cl_context context = NULL;
cl_program program = NULL;
cl_device_id device;
cl_kernel knl_memRdData;
cl_kernel knl_memRdWeight;
cl_kernel knl_controller;
cl_kernel knl_memWrite;

cl_command_queue que_memRdData;
cl_command_queue que_memRdWeight;
cl_command_queue que_controller;
cl_command_queue que_memWrite;

cl_mem config_buf;
cl_mem bottom0_buf;
cl_mem bottom1_buf;
cl_mem weights_buf;
cl_mem bias_buf;

DTYPE *weights;
DTYPE *biases;
DTYPE *image;
DTYPE *data_init;

unsigned layer_config_original[LAYER_NUM][NUM_CONFIG_ITEM];

void loadImageToBuffer(int num);
int  prepare();
void printCurrentTime();
void cleanup();

int main(int argc, char** argv)
{
	cl_int status;

	unsigned int weight_buf_size;
	unsigned int bias_buf_size;
	unsigned int pic_num = 1;

	if (argc != 2){
		printf("Error: wrong commad format, usage:\n");
		printf("%s <binaryfile>\n", argv[0]);
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
	device = *(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("  Device %d: %s\n", i, getDeviceName(device).c_str());
		displayDeviceInfo(device);
	}

	// Create the context.
	context = clCreateContext(NULL, num_devices, &device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create Program Objects
	char *kernel_file_name=argv[1];

	// Create the program for all device. All devices execute the same kernel.
	program = createProgramFromFile(context, (const char *) kernel_file_name, &device, num_devices);

	// Prepare compute data
	status = prepare();
	if(status == 1) {
		printf("Allocate memory for data and weights failed !!!\n");
		return false;
	}

	// Command queue
	
	printf ("[INFO] Creating the command queues\n");
	que_memRdData = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue for memReadData");
	que_memRdWeight = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue for memRdWeight");
	que_controller = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue for controller");
	que_memWrite = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue for memWrite");

	// Kernel
	
	printf ("[INFO] Creating the kernels\n");
	knl_memRdData = clCreateKernel(program, knl_name_memRdData, &status);
	checkError(status, "Failed to create memRdData kernel");

	knl_memRdWeight = clCreateKernel(program, knl_name_memRdWeight, &status);
	checkError(status, "Failed to create memRdWeight kernel");

	knl_controller = clCreateKernel(program, knl_name_controller, &status);
	checkError(status, "Failed to create controller kernel");

	knl_memWrite = clCreateKernel(program, knl_name_memWrite, &status);
	checkError(status, "Failed to create memWrite kernel");

	// Here we should calculate how many items we will have in the whole weight buffer.
	// For each layer, we have the total number of output cannels.
	// Each output channels, is a brick which is of size
	// => num_input_channels * weight_h * W_VEC
	// as you can see, the weight_w is replaced by the W_VEC, because we are using the
	// winograd transformation.
	weight_buf_size = 0;
	for (int layer = 0; layer < LAYER_NUM; layer++) {
		weight_buf_size += (layer_config[layer][weight_m]*layer_config[layer][weight_n]*layer_config[layer][weight_h]*W_VEC);
	}
	printf ("[INFO] Creating the weight buffer\n");
	weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_buf_size * sizeof(DTYPE), NULL, &status);
	checkError(status, "Failed to create buffer for weights in layer");

	// The total number of biases, is equal to the total number of layers,
	// multiplied by the number of output channels of the each layer
	bias_buf_size = 0;
	for (int layer = 0; layer < LAYER_NUM; layer++) {
		bias_buf_size += layer_config[layer][weight_m];
	}
	printf ("[INFO] Creating the bias buffer\n");
	bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_buf_size * sizeof(DTYPE), NULL, &status);
	checkError(status, "Failed to create buffer for bias in layer");

	// Initializing all weights buffers, blocking write is used
	//

	printf ("[INFO] weight_buf_size=%d, WEIGHTS_FILE_SIZE=%d\n", weight_buf_size, WEIGHTS_FILE_SIZE);

	printf ("[INFO] Enqueueing the weight buffer to weight queue\n");
	status = clEnqueueWriteBuffer(que_memRdWeight, weights_buf, CL_TRUE, 0, weight_buf_size * sizeof(DTYPE), weights, 0, NULL, NULL);
	checkError(status, "Failed to transfer weight");

	printf ("[INFO] biases_buf_size=%d, BIASES_FILE_SIZE=%d\n", bias_buf_size, BIASES_FILE_SIZE);	

	printf ("[INFO] Enqueueing the bias buffer to bias queue\n");
	status = clEnqueueWriteBuffer(que_memRdWeight, bias_buf, CL_TRUE, 0, bias_buf_size * sizeof(DTYPE), biases, 0, NULL, NULL);
	checkError(status, "Failed to transfer bias");

	// First buffer
	printf ("[INFO] Creating the bottom0 buffer\n");
	bottom0_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, IN_BUF_SIZE * sizeof(DTYPE), NULL, &status);
	checkError(status, "Failed to create buffer for data in layer");

	// Second buffer
	printf ("[INFO] Creating the bottom1 buffer\n");
	bottom1_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, OUT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
	checkError(status, "Failed to create buffer for output");

	//printf ("[INFO] Enqueueing the bottom0 buffer\n");
	//status = clEnqueueWriteBuffer(que_memRdData, bottom0_buf, CL_TRUE, 0, IN_BUF_SIZE * sizeof(DTYPE), NULL, 0, NULL, NULL);
	//checkError(status, "Failed to transfer bottom0");

	//printf ("[INFO] Enqueueing the bottom1 buffer\n");
	//status = clEnqueueWriteBuffer(que_memRdData, bottom1_buf, CL_TRUE, 0, OUT_BUF_SIZE * sizeof(DTYPE), NULL, 0, NULL, NULL);
	//checkError(status, "Failed to transfer bottom1");

	printf ("[INFO] Creating the config buffer with size %d\n", sizeof(fpga_configuration) * LAYER_NUM);
	config_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(fpga_configuration) * LAYER_NUM, NULL, &status);
	checkError(status, "Failed to create buffer for config");

	fpga_configuration config[LAYER_NUM];

	printf ("[INFO] Setting the configurations per layer\n");
	for (int layer = 0; layer < LAYER_NUM; layer++) {
		config[layer].layer_type = layer_config[layer][layer_type];
		config[layer].data_w = layer_config[layer][data_w];
		config[layer].data_h = layer_config[layer][data_h];
		config[layer].weight_w = layer_config[layer][weight_w];
		config[layer].weight_h = layer_config[layer][weight_h];
		config[layer].weight_n = layer_config[layer][weight_n];
		config[layer].weight_m = layer_config[layer][weight_m];
		config[layer].memrd_src = layer_config[layer][memrd_src];
		config[layer].conv_x = layer_config[layer][conv_x];
		config[layer].conv_y = layer_config[layer][conv_y];
		config[layer].conv_z = layer_config[layer][conv_z];
		config[layer].conv_stride = layer_config[layer][conv_stride];
		config[layer].conv_padding = layer_config[layer][conv_padding];
		config[layer].conv_split = layer_config[layer][conv_split];
		config[layer].conv_relu = layer_config[layer][conv_relu];
		config[layer].pool_on = layer_config[layer][pool_on];
		config[layer].pool_x = layer_config[layer][pool_x];
		config[layer].pool_y = layer_config[layer][pool_y];
		config[layer].pool_z = layer_config[layer][pool_z];
		config[layer].pool_size = layer_config[layer][pool_size];
		config[layer].conv_stride = layer_config[layer][pool_stride];
		config[layer].lrn_on = layer_config[layer][lrn_on];
		config[layer].memwr_dst = layer_config[layer][memwr_dst];

		printf ("[INFO] layer_type: %d, data_w: %d, data_h: %d, weight_w: %d, weight_h: %d, weight_n: %d, weight_m: %d, memrd_src: %d, conv_x: %d, conv_y: %d, conv_z: %d, conv_stride: %d, conv_padding: %d, conv_split: %d, conv_relu: %d, pool_on: %d, pool_x: %d, pool_y: %d, pool_z: %d, pool_size: %d, conv_stride: %d, lrn_on: %d, memwr_dst: %d\n", layer_config[layer][layer_type], layer_config[layer][data_w], layer_config[layer][data_h], layer_config[layer][weight_w], layer_config[layer][weight_h], layer_config[layer][weight_n], layer_config[layer][weight_m], layer_config[layer][memrd_src], layer_config[layer][conv_x], layer_config[layer][conv_y], layer_config[layer][conv_z], layer_config[layer][conv_stride], layer_config[layer][conv_padding], layer_config[layer][conv_split], layer_config[layer][conv_relu], layer_config[layer][pool_on], layer_config[layer][pool_x], layer_config[layer][pool_y], layer_config[layer][pool_z], layer_config[layer][pool_size], layer_config[layer][pool_stride], layer_config[layer][lrn_on], layer_config[layer][memwr_dst]);
		
		int w_vec = W_VEC;
		config[layer].num_bricks = (layer_config[layer][data_h]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_h]+1)*((layer_config[layer][data_w]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_w])/(W_VEC-layer_config[layer][weight_w]+1) + 1);
		printf ("[INFO] w_vec: %d\n", w_vec);
		printf ("[INFO] data_w: %d\n", layer_config[layer][data_w]);
		printf ("[INFO] data_h: %d\n", layer_config[layer][data_h]);
		printf ("[INFO] conv_padding: %d\n", layer_config[layer][conv_padding]);
		printf ("[INFO] weight_w: %d\n", layer_config[layer][weight_w]);	
		printf ("[INFO] Some #1: %d\n", layer_config[layer][data_h]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_h]+1);
		printf ("[INFO] first part: %d\n", (layer_config[layer][data_w]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_w]));
		printf ("[INFO] second part: %d\n", (w_vec-layer_config[layer][weight_w]+1));
		printf ("[INFO] Some #2: %d\n", ((layer_config[layer][data_w]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_w])/(w_vec-layer_config[layer][weight_w]+1)) + 1);
		// printf ("[INFO] Some #1: %d, Some #2: %d, data_w: %d, conv_padding: %d, w_vec: %d, weight_w: %d\n", layer_config[layer][data_h]+2*layer_config[layer][conv_padding]-layer_config[layer][weight_h]+1, ceil((layer_config[layer][data_w]+2*layer_config[layer][conv_padding]-W_VEC)/(W_VEC-layer_config[layer][weight_w]+1)), layer_config[layer][data_w], layer_config[layer][conv_padding], w_vec, layer_config[layer][weight_w]);
	}

	printf ("[INFO] Enqueueing the config buffer to controller queue\n");
	status = clEnqueueWriteBuffer(que_controller, config_buf, CL_TRUE, 0, sizeof(fpga_configuration) * LAYER_NUM, config, 0, NULL, NULL);
	checkError(status, "Failed to transfer config");

	// Execute the kernel
	cl_event memRdData_event;
	cl_event memRdWeight_event;
	cl_event memWrite_event;
	cl_event controller_event;

	// Recorde the excution time of each operation
	cl_ulong memRdData_time;
	cl_ulong memRdWeight_time;
	cl_ulong memWrite_time;
	cl_ulong controller_time;
	
	for (int iter = 0; iter < 1; iter++) {
	
		printf ("[INFO] Iteration number #%d\n", iter);
	
		loadImageToBuffer(pic_num);
		printCurrentTime();
		unsigned argi = 0;
		char layer_num = LAYER_NUM;

		// Setting the arguments for the controller
		argi = 0;

		printf ("[INFO] Setting kernel arguments for the controller\n ");
		status = clSetKernelArg(knl_controller, argi++, sizeof(cl_char), &layer_num);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller, argi++, sizeof(cl_char), &precision_config[0][frac_w]);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller, argi++, sizeof(cl_char), &precision_config[0][frac_din]);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller, argi++, sizeof(cl_char), &precision_config[0][frac_dout]);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		status = clSetKernelArg(knl_controller, argi++, sizeof(cl_mem), &config_buf);
		checkError(status, "Failed to set argument %d of kernel controller", argi-1);

		// Setting the arguments for the memory read data module
		argi = 0;
		char config_size = LAYER_NUM;

		printf ("[INFO] Setting kernel arguments for the memRdData\n");
		status = clSetKernelArg(knl_memRdData, argi++, sizeof(cl_char), &config_size);
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		status = clSetKernelArg(knl_memRdData, argi++, sizeof(cl_mem), &bottom0_buf);
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		status = clSetKernelArg(knl_memRdData, argi++, sizeof(cl_mem), &bottom1_buf);
		checkError(status, "Failed to set argument %d of kernel memory read data", argi-1);

		argi = 0;

		printf ("[INFO] Setting kernel arguments for the memRdWeight\n");
		status = clSetKernelArg(knl_memRdWeight, argi++, sizeof(cl_char), &config_size);
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		status = clSetKernelArg(knl_memRdWeight, argi++, sizeof(cl_mem), &weights_buf);
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		status = clSetKernelArg(knl_memRdWeight, argi++, sizeof(cl_mem), &bias_buf);
		checkError(status, "Failed to set argument %d of kernel memory read weight", argi-1);

		argi = 0;

		printf ("[INFO] Setting kernel arguments for the memWrite\n");
		status = clSetKernelArg(knl_memWrite, argi++, sizeof(cl_char), &config_size);
		checkError(status, "Failed to set argument %d of kernel memory write");

		status = clSetKernelArg(knl_memWrite, argi++, sizeof(cl_mem), &bottom0_buf);
		checkError(status, "Failed to set argument %d of kernel memory write");

		status = clSetKernelArg(knl_memWrite, argi++, sizeof(cl_mem), &bottom1_buf);
		checkError(status, "Failed to set argument %d of kernel memory write");

		// Enqueueing kernels
		printf ("[INFO] Enqueuing tasks [controller,memRdData,memRdWeight,memWrite]\n");
		status = clEnqueueTask(que_controller, knl_controller, 0, NULL, &controller_event);
		checkError(status, "Failed to launch kernel controller");

		status = clEnqueueTask(que_memRdData, knl_memRdData, 0, NULL, &memRdData_event);
		checkError(status, "Failed to launch kernel memory read data");

		status = clEnqueueTask(que_memRdWeight, knl_memRdWeight, 0, NULL, &memRdWeight_event);
		checkError(status, "Failed to launch kernel memory read weight");

		status = clEnqueueTask(que_memWrite, knl_memWrite, 0, NULL, &memWrite_event);
		checkError(status, "Failed to launch kernel write");

		// Waiting for the events
		status = clWaitForEvents(1, &controller_event);
		checkError(status, "Failed to wait for the controller kernel\n");
		printf ("[INFO] Done with the controller!\n");

		status = clWaitForEvents(1, &memRdData_event);
		checkError(status, "Failed to wait for the memRdData kernel\n");
		printf ("[INFO] Done with the memReadData!\n");
		
		status = clWaitForEvents(1, &memRdWeight_event);
		checkError(status, "Failed to wait for the memRdWeight kernel\n");
		printf ("[INFO] Done with the memReadWeight!\n");

		status = clWaitForEvents(1, &memWrite_event);
		checkError(status, "Failed to wait for the memWrite kernel\n");
		printf ("[INFO] Done with memWrite!\n");

		printf ("[INFO] Calculating kernel runtime\n");
		memRdData_time = getKernelStartEndTime(memRdData_event, "memRd");
		memRdWeight_time = getKernelStartEndTime(memRdWeight_event, "conv");
		memWrite_time = getKernelStartEndTime(memWrite_event, "memWr");
		controller_time = getKernelStartEndTime(controller_event, "lrn");

		// Must release event object to avoid performance degeneration !!!

		printf ("[INFO] Releasing events\n");
		status = clReleaseEvent(memRdData_event);
		checkError(status, "Failed to release mem read data event object");
		status = clReleaseEvent(memRdWeight_event);
		checkError(status, "Failed to release mem read weight event object");
		status = clReleaseEvent(memWrite_event);
		checkError(status, "Failed to release mem write event object");
		status = clReleaseEvent(controller_event);
		checkError(status, "Failed to release controller event object");

		printCurrentTime();

	} // end of iterations

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
	status = clEnqueueWriteBuffer(que_memRdData, bottom0_buf, CL_TRUE, 0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
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
				|| input_config[image_n] != layer_config_original[ll][data_n] || input_config[image_n] != layer_config_original[ll][weight_n]){
					printf("Error: incorrect layer configuration for layer-%d !!!\n", ll+1);
					//return 1;
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
				return 1;
			}
			if((layer_config_original[ll][data_n]!=layer_config_original[ll-1][conv_z])){
				printf("\nError: incorrect setting of convolution input/output size for layer-%d!!!\n", ll+1);
				return 1;
			}
		}
		if((layer_config_original[ll][conv_x]!=(layer_config_original[ll][data_w]-layer_config_original[ll][weight_w]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
			|| (layer_config_original[ll][conv_y]!=(layer_config_original[ll][data_h]-layer_config_original[ll][weight_h]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
		    || (layer_config_original[ll][conv_z]!=layer_config_original[ll][weight_m])){
			printf("\nError: incorrect setting of convolution output size or filter params for layer-%d!!!\n", ll+1);
			return 1;
		}
		if(layer_config_original[ll][pool_on] && ((layer_config_original[ll][pool_x]!=(layer_config_original[ll][conv_x]-layer_config_original[ll][pool_size])/layer_config_original[ll][pool_stride]+1)
			|| (layer_config_original[ll][pool_y]!=(layer_config_original[ll][conv_y]-layer_config_original[ll][pool_size])/layer_config_original[ll][pool_stride]+1)
		    || (layer_config_original[ll][pool_z]!=layer_config_original[ll][conv_z]))){
			printf("\nError: incorrect setting of pooling input/output size for layer-%d!!!\n", ll+1);
			return 1;
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
	// padding the input RGB image with extra number of zeros channels, so that data_n/weight_n is divisible by VEC_SIZE
	layer_config[0][weight_n] = ceil((float)layer_config[0][weight_n]/VEC_SIZE)*VEC_SIZE;
	printf ("[INFO] weight_n is changed to %d\n", layer_config[0][weight_n]);
	layer_config[0][data_n] = layer_config[0][weight_n];

	data_init   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n], DMA_ALIGNMENT);
	memset(data_init, 0, sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]);// fill non-RGB dims with 0

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

	return 0;
}

// Release all memory resources here
void cleanup()
{

	// Release the opencl runtime resource allocated
	for(unsigned i = 0; i < num_devices; ++i) {

		// Killing the kernels
		if(knl_memRdData) {
			clReleaseKernel(knl_memRdData);
		}
		if(knl_memRdWeight) {
			clReleaseKernel(knl_memRdWeight);
		}
		if(knl_controller) {
			clReleaseKernel(knl_controller);
		}
		if(knl_memWrite) {
			clReleaseKernel(knl_memWrite);
		}

		// Killing all the queues
		if(que_memRdData) {
			clReleaseCommandQueue(que_memRdData);
		}
		if(que_memRdWeight) {
			clReleaseCommandQueue(que_memRdWeight);
		}
		if(que_controller) {
			clReleaseCommandQueue(que_controller);
		}
		if(que_memWrite) {
			clReleaseCommandQueue(que_memWrite);
		}

		// Killing all the buffers
		if(config_buf) {
			clReleaseMemObject(config_buf);
		}
		if(bottom0_buf) {
			clReleaseMemObject(bottom0_buf);
		}
		if(bottom1_buf) {
			clReleaseMemObject(bottom1_buf);
		}
		if(weights_buf) {
			clReleaseMemObject(weights_buf);
		}
		if(bias_buf) {
			clReleaseMemObject(bias_buf);
		}
	}

	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}

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
