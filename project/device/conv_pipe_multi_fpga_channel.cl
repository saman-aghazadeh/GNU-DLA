/*
 * ------------------------------------------------------
 *
 *   PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs
 *
 * ------------------------------------------------------
 * Filename:
 *   - conv_pipe_multi_fpga.cl
 *
 * Author(s):
 *   - Saman Biookaghazadeh, sbiookag@asu.edu
 *   - Dong Wang, wangdong@m.bjtu.edu.cn
 *
 * History:
 *   - cascaded multiple kernels, while mapping them 
 *     on multiiple FPGAs. FPGAs are talking to each
 *     through the IO channel.
 *   - v1.3 Win-Buffer-Based Implementation
 * ------------------------------------
 *
 *   Copyright (C) 2019, Arizona State University. All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 */

#define USE_ROM 

// The following macros are used for debug
//#define DEBUG_MEMRD
//#define DEBUG_CONV
//#define DEBUG_POOL
//#define DEBUG_MEMWR
//#define DEBUG_LRN
//#define DEBUG_LRN_OUT

#include "hw_param.cl"
#include "rtl_lib.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable


// Define the precision of the data-path
typedef char DPTYPE;
typedef int  MACTYPE;

// Vectorized data type
typedef struct {
   DPTYPE data[VEC_SIZE];
} lane_data;

// Combined vec-data type from multiple lane
typedef struct {
   lane_data lane[LANE_NUM];
} channel_vec;

// Combined scalar data type from multiple lane
typedef struct {
   DPTYPE lane[LANE_NUM];
} channel_scal;

channel channel_vec    data_ch    __attribute__((depth(0)));
channel channel_vec    weight_ch  __attribute__((depth(0)));
channel channel_scal   bias_ch    __attribute__((depth(8)));
channel channel_scal   conv_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_scal   pool_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_scal   bypass_ch  __attribute__((depth(CHN_DEPTH)));
channel ulong4      ser_ch     __attribute__((depth(4))) __attribute__((io("kernel_output_ch1")));
channel ulong4      deser_ch   __attribute__((depth(4))) __attribute__((io("kernel_input_ch0")));


// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(lane_data input, lane_data weights)
{
	MACTYPE output = MASK_MULT & CZERO;
	
	#pragma unroll
	for(int i=0; i<VEC_SIZE/4; i++){
		output += MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4], input.data[i*4+1], weights.data[i*4+1], input.data[i*4+2], weights.data[i*4+2], input.data[i*4+3], weights.data[i*4+3]);
	}
	return output;
}

DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in)
{
	DPTYPE max_value;
	
	if(a_in >= b_in)
		max_value = a_in;
	else
		max_value = b_in;
	
	return max_value;

}


#include "multi_fpga_channel/L.cl"
