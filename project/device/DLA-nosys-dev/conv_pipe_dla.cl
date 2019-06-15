/*
 * ------------------------------------------------------
 *
 *   GNU-DLA: An OpenCL-Based FPGA Accelerator for CNNs
 *
 * ------------------------------------------------------
 * Filename:
 *   - conv_pipe_dla.cl
 *
 * Author(s):
 *   - Saman Biookaghazadeh, sbiookag@asu.edu
 * ------------------------------------
 *
 *   Copyright (C) 2019, CIDSE,
 *   Arizona State University. All rights reserved.
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

#include "hw_param.cl"
#include "ihc_apint.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

// Define the precision of the data-path
typedef int8_t DPTYPE;
typedef int16_t  MACTYPE;

// Vectorized data type
typedef struct {
   DPTYPE data[VEC_SIZE];
} lane_data;

typedef struct {
	MACTYPE w_data[W_VEC];
} w_data;

typedef struct {
	MACTYPE w_inv_data[W_INV_VEC];
} w_inv_data;

// Combined vec-data type from multiple lane
typedef struct {
   lane_data lane[LANE_NUM];
} channel_vec;

// Combined scalar data type from multiple lane
typedef struct {
	DPTYPE lane[LANE_NUM];
} channel_scal;

// Combine vec-data type for multiple columns
typedef struct {
	lane_data cols[W_VEC];
} lane_cols;

typedef struct {
	w_data cols[LANE_NUM];
} channel_cols;

typedef struct {
	channel_scal cols[W_INV_VEC];
} inv_rows;

typedef struct {
	DPTYPE bias[LANE_NUM];
} bias_DPTYPE;

typedef struct {
	lane_cols weight[LANE_NUM];
} weight_lane_cols;

//
typedef struct {
	uint conv_loop_cnt;
	int frac_w;
	int frac_din;
	int frac_dout;
	int num_weight_plates;
	int out_ch_per_pe;
	int num_bricks;
} instruction;

// configuration of the layer
typedef struct {
	int layer_type; // 0 -> conv, 1 -> fc
	int data_w, data_h, weight_w, weight_h, weight_n, weight_m, bias_size; 
	int memrd_src;
	int conv_x, conv_y, conv_z, conv_stride, conv_padding, conv_split, conv_relu;
	int pool_on, pool_x, pool_y, pool_z, pool_size, pool_stride;
	int lrn_on; // lrn on/off control
	int memwr_dst; // 0 -> data_buf, 1 -> output_buf, 2 -> fc_1_buffer, 3 -> fc_2_buffer
	int num_bricks;
} configuration;

// required config for the memRdData
typedef struct {
	int layer_type;
	int data_w, data_h;
	int weight_m;
	int weight_n;
	int weight_h;
	int weight_w;
	int conv_padding;
} memrd_data_configuration;

// required config for the memRdWeight
typedef struct {
	int weight_m;
	int weight_n;
	int weight_h;
	int weight_w;
} memrd_weight_configuration;

// required config for the memWr
typedef struct {
	int conv_x;
	int conv_y;
	int conv_z;
	int weight_w;
} memwr_configuration;


channel lane_cols					chain_data_channels[LANE_NUM];
channel lane_cols					winograd_transform_channels;
channel inv_rows					winograd_inv_transform_channels;
channel lane_cols					weight_channels[LANE_NUM];
channel DPTYPE						bias_channels[LANE_NUM];
// channel channel_cols				chain_output_channels[LANE_NUM+1];

channel memrd_data_configuration	memrd_data_configuration_channel;
channel memrd_weight_configuration	memrd_weight_configuration_channel;
channel memwr_configuration 		memwr_configuration_channel;
channel instruction					chain_instruction_channels[LANE_NUM+1];

// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(lane_data input, lane_data weights)
{
	MACTYPE output = MASK_MULT & CZERO;		

	#pragma unroll	
	for (int i = 0; i < VEC_SIZE; i++) {
		output += (input.data[i] * weights.data[i]);
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
#include "PE_header.cl"
#include "controller.cl"
#include "mem_read_data.cl"
#include "mem_read_weight.cl"
#include "PE.cl"
#include "mem_write.cl"
#include "winograd_inv_transform.cl"
#include "winograd_transform.cl"
