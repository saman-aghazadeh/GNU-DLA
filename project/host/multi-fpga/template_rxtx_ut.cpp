// Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This host program executes a vector addition kernel to perform:
//  C = A + B
// where A, B and C are vectors with N elements.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device types
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
//#include "AOCLUtils/aocl_utils.h"
#include "AOCLUtils/aocl_utils.h"
#include <byteswap.h>
#include "fpgaStreamDrv.h"

using namespace aocl_utils;
using namespace fpga;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> algo_kernel; // num_devices elements
scoped_array<cl_kernel> data_rdma_kernel; // num_devices elements
scoped_array<cl_kernel> ctrl_rdma_kernel; // num_devices elements
scoped_array<cl_kernel> tx_rdma_kernel; // num_devices elements
scoped_array<cl_kernel> tx_io_kernel; // num_devices elements
scoped_array<cl_kernel> rx_io_kernel; // num_devices elements

scoped_array<cl_mem> o_params_buf; // num_devices elements

// Problem data.
unsigned long N = 64; // problem size
scoped_array<scoped_aligned_ptr<unsigned long> > o_params; // num_devices elements

scoped_array<unsigned> n_per_device; // num_devices elements
scoped_array<unsigned> modulo; // num_devices elements
unsigned long input_modulo=16;
unsigned long red=4;
unsigned long rep=2;

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

void close_rx_stream_handle(fpga_meta hFPGA)
{
  close_stream(hFPGA);
}

fpga_meta setup_rx_regmap()
{

     fpga_meta fpga1_meta, fpga2_meta;
     int nStatus =0;
     int num_inputs = 4;
     bool reverse = false;

     string src_mac({(char)0x14, (char)0x18, (char) 0x77, (char)0x33, (char)0xB0, (char) 0x6B});
     string fpga1_port8({(char)0x00, (char)0x0C, (char)0xD7, (char)0x00, (char)0x2C, (char)0x10});
     string packet_type({(char)0xAA, (char)0xAA});
     unsigned long addr, value;
     //setup comm
     nStatus = setup_comm(src_mac, fpga1_port8, packet_type, &fpga1_meta);

    for(int i=0; i <  num_inputs; i++)
    {
      //data limit
      addr =  i;
      value = 16382;  //sets the algorithm trigger
      nStatus = write_region(fpga1_meta, SYS_CTRL, addr, value, reverse);
      //dst address
      addr += 4;
      value = 0x1814102C00D70C00;
      nStatus = write_region(fpga1_meta, SYS_CTRL, addr, value, reverse);
      //src address
      addr += 4;
      value = 0x0000AAAA6BB03377;
      nStatus = write_region(fpga1_meta, SYS_CTRL, addr, value, reverse);
      //input number
      addr += 4;
      value = i;
      nStatus = write_region(fpga1_meta, SYS_CTRL, addr, value, reverse);
   
    }

    return fpga1_meta;

}

int send_dummy_stream_data(fpga_meta hFPGA)
{
  unsigned long size = 16384;
  unsigned long dummy_data[size];
  int num_inputs = 4;
  int nStatus = 0;

  //set header data
  dummy_data[0] = DATA_CTRL; //classification
  dummy_data[2] = 0;         //starting index
 
  for(int i=0; i < num_inputs; i++)
  {
    //fill dummy data
    for(int j=2; j < (size); j++)
    {  dummy_data[j] = 0xA0 + i;}
    
    //channel ID
    dummy_data[1]= i;

    //send data
    nStatus = write_stream_data(hFPGA, "", (const void *) dummy_data, size*sizeof(unsigned long));
    //sleep(2);
  }

 return 0; 
}

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Optional argument to specify the problem size.
  if(options.has("n")) {
    N = options.get<unsigned>("n");
  }
  if(options.has("mod")) {
    input_modulo = options.get<unsigned>("mod");
  }
  if(options.has("red")) {
    red = options.get<unsigned>("red");
  }
  if(options.has("rep")) {
    rep = options.get<unsigned>("rep");
  }

  printf("\nproblem_size= %lx, input modulo = %lx\n", N, input_modulo);

  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  printf("Overriding number of devices to 1\n");
  for(int i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  num_devices=1;
  // Create the context.
  context = clCreateContext(NULL, num_devices, &device[1], &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("template_algo", device[1]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device[1], num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  int num_of_queues = 7;
  queue.reset(num_of_queues);
  //RX
  algo_kernel.reset(1);
  data_rdma_kernel.reset(1);
  ctrl_rdma_kernel.reset(1);
  rx_io_kernel.reset(1);
  //TX
  tx_io_kernel.reset(1);
  tx_rdma_kernel.reset(1);

  n_per_device.reset(1);
  modulo.reset(num_devices);
  o_params_buf.reset(10); //number of inputs buffers to kernels (not used in test case)

  // Command queue.
  queue[0] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 0");
  queue[1] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 1");
  queue[2] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 2");
  queue[3] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 3");
  queue[4] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 4");
  queue[5] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 5");
  queue[6] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue 6");

  // Kernel.
  //normal kernel
  //RX
  const char *algo_name        = "algo";
  const char *data_rdma_engine = "rx_data_arb";
  const char *ctrl_rdma_engine = "rx_ctrl_arb";
  const char *rx_io_channel    = "rx_io_channel";
  //TX
  const char *tx_rdma_engine   = "tx_rdma_arb";
  const char *tx_io_channel    = "tx_io_channel";

  ////////////////////////MAIN ALGO///////////////////////////////////////
  algo_kernel[0] = clCreateKernel(program, algo_name, &status);
  checkError(status, "Failed to create algo kernel");

  ///////////////////////////////RX////////////////////////////////////////
  data_rdma_kernel[0] = clCreateKernel(program, data_rdma_engine, &status);
  checkError(status, "Failed to create rx data rdma  kernel");

  ctrl_rdma_kernel[0] = clCreateKernel(program, ctrl_rdma_engine, &status);
  checkError(status, "Failed to create rx ctrl rdma kernel");

  rx_io_kernel[0] = clCreateKernel(program, rx_io_channel, &status);
  checkError(status, "Failed to create rx IO kernel");

  ///////////////////////////////TX////////////////////////////////////////
  tx_io_kernel[0] = clCreateKernel(program, tx_io_channel, &status);
  checkError(status, "Failed to create tx IO  kernel");

  tx_rdma_kernel[0] = clCreateKernel(program, tx_rdma_engine, &status);
  checkError(status, "Failed to create tx rdma kernel");
  /////////////////////////////////////////////////////////////////////////

  // Input buffers.
  for(int j=0; j < 10; j++)
  {
    //o_params_buf[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
    o_params_buf[j] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        16384 * sizeof(unsigned long), NULL, &status);
    checkError(status, "Failed to create buffer for input A");
  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  //o_params.reset(5);

}

void run() {
  cl_int status;

  const double start_time = getCurrentTimestamp();
  cl_event write_event[7];

  unsigned argi = 0;

  //setting algo parameters
  for(int j=1; j < 10; j++)
  {
    status = clSetKernelArg(algo_kernel[0], (j-1), sizeof(cl_mem), &o_params_buf[j]);
    checkError(status, "Failed to set argument %d in algo_kernel", (j-1));
  }

  ////////////////////////////////////////////////////////////////////////

  //sys config
  status = clSetKernelArg(data_rdma_kernel[0], 0, sizeof(cl_mem), &o_params_buf[0]);
  checkError(status, "Failed to set argument %d in algo_kernel", 0);
  //setting algo parameters
  for(int j=2; j < 6; j++)
  {
    status = clSetKernelArg(data_rdma_kernel[0], (j-1), sizeof(cl_mem), &o_params_buf[j]);
    checkError(status, "Failed to set argument %d in algo_kernel", (j-1));
  }

  ////////////////////////////////////////////////////////////////////////
   
  for(int j=0; j < 2; j++)
  {
    status = clSetKernelArg(ctrl_rdma_kernel[0], j, sizeof(cl_mem), &o_params_buf[j]);
    checkError(status, "Failed to set argument %d in algo_kernel", j);
  }
  //////////////////////////////////////////////////////////////////////// 
  //ADDING TX SECTION
  // Set kernel arguments.
  status = clSetKernelArg(tx_rdma_kernel[0], 0, sizeof(cl_mem), &o_params_buf[0]);
  checkError(status, "Failed to set argument %d in algo_kernel", 0);
  for(int j=6; j < 10; j++)
  {
    status = clSetKernelArg(tx_rdma_kernel[0], j-5, sizeof(cl_mem), &o_params_buf[j]);
    checkError(status, "Failed to set argument %d in rdma_kernel", j-1);
  }



 //////////////////////////////////////////////////////////////////////// 
  //add ghost kernel single work itte

  printf("\n\nStarting IO kernel\n\n");
  status = clEnqueueTask(queue[0], rx_io_kernel[0], 0, NULL, &write_event[0]);
  printf("status = %i\n", status);
  checkError(status, "Failed to start io kernel\n");
 
  printf("\n\nStarting IO kernel\n\n");
  status = clEnqueueTask(queue[1], tx_io_kernel[0], 0, NULL, &write_event[1]);
  printf("status = %i\n", status);
  checkError(status, "Failed to start io kernel\n");

  //add kernel single work itte
  printf("\n\nStarting Data RDMA kernel\n\n");
  status = clEnqueueTask(queue[2], data_rdma_kernel[0], 0, NULL, &write_event[2]);
  printf("status = %i\n", status);
  checkError(status, "Failed to start rdma kernel\n");
  
  //add kernel single work itte
  printf("\n\nStarting Ctrl RDMA kernel\n\n");
  status = clEnqueueTask(queue[3], ctrl_rdma_kernel[0], 0, NULL, &write_event[3]);
  printf("status = %i\n", status);
  checkError(status, "Failed to start rdma kernel\n");

  //add kernel single work itte
  printf("\n\nStarting tx RDMA kernel\n\n");
  status = clEnqueueTask(queue[4], tx_rdma_kernel[0], 0, NULL, &write_event[4]);
  printf("status = %i\n", status);
  checkError(status, "Failed to start tx rdma kernel\n");


  printf("\n\nStarting algo kernel\n\n");
  status = clEnqueueTask(queue[5], algo_kernel[0], 0, NULL, &write_event[5]);
  printf("status = %i\n", status);
  checkError(status, "Failed to start algo kernel\n");
  
  //Get the profile data before the kernel completes
  //clGetProfileInfoIntelFPGA(write_event[5]);  
  //clWaitForEvents(1, &write_event[5]); 
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  fpga_meta hFPGA; 
  //setting up control registers
  sleep(5);
  hFPGA = setup_rx_regmap();
  sleep(5);

  printf("Start sending dummy data \n");
  send_dummy_stream_data(hFPGA);
  printf("Finished sending dummy data \n");
  printf( "What time is it? \n");

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  clWaitForEvents(1, &write_event[5]); 
  printf("Peanut Butter & Jelly Time\n");

  //CONFIG REGION
  //very llose synchronization

  //get sys_config data
  scoped_array<scoped_aligned_ptr<ulong> > output; // num_devices elements

  output.reset(10);
  for(int i=0; i < 10; i++)
  { 
    output[i].reset(4096);
    for(int j=0; j <4096; j++)
      output[i][j] = 0;
  }
  
  printf("\nReading System Config\n");
  status = clEnqueueReadBuffer(queue[6], o_params_buf[0], CL_FALSE,
         0, 16 * sizeof(ulong), output[0], 0, NULL, &write_event[6]);
  clWaitForEvents(1, &write_event[6]); 

  for(int i=0; i < 32; i++)
    printf("%i) %16x\n",i,output[0][i]); 

  for(int i=2; i < 10; i++)
  {
    if( i < 6)
      printf("\nReading Input Parameter %i:\n",i-1);
    else
      printf("\nReading Output Parameter %i:\n",i-5);
    status = clEnqueueReadBuffer(queue[6], o_params_buf[i], CL_FALSE,
           0, 512 * sizeof(ulong), output[i], 1, &write_event[6], &write_event[6]);
    clWaitForEvents(1, &write_event[6]); 
    //show results
    for(int j=0; j < 512; j++)
      printf("%i)%i) %16x\n",i,j,output[i][j]);
  }


  printf("Closing socket interface");
  close_rx_stream_handle(hFPGA);
}

// Free the resources allocated during initialization
void cleanup() {
  /*for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }
  }
*/
  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

