/*
 * ------------------------------------------------------
 *
 *   PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs
 *
 * ------------------------------------------------------
 * Filename:
 *   - conv_pipe.cl
 *
 * Author(s):
 *   - Dong Wang, wangdong@m.bjtu.edu.cn
 *
 * History:
 *   - v1.3 Win-Buffer-Based Implementation
 * ------------------------------------
 *
 *   Copyright (C) 2016, Institute of Information Science,
 *   Beijing Jiaotong University. All rights reserved.
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


#ifndef EMULATE
channel lane_data      data_ch    __attribute__((depth(0)));
channel channel_vec    weight_ch  __attribute__((depth(0)));
channel channel_scal   bias_ch    __attribute__((depth(8)));
//channel channel_scal   conv_ch    __attribute__((depth(CHN_DEPTH)));
//channel channel_scal   pool_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_scal   bypass_ch  __attribute__((depth(CHN_DEPTH)));
#else
channel channel_vec    data_ch_write    __attribute__((io("dataCh")))   __attribute__((depth(0)));
channel channel_vec    data_ch_read     __attribute__((io("dataCh")))   __attribute__((depth(0)));

channel channel_vec    weight_ch_write  __attribute__((io("weightCh"))) __attribute__((depth(0)));
channel channel_vec    weight_ch_read   __attribute__((io("weightCh"))) __attribute__((depth(0)));

channel channel_scal   bias_ch_write    __attribute__((io("biasCh")))   __attribute__((depth(8)));
channel channel_scal   bias_ch_read     __attribute__((io("biasCh")))   __attribute__((depth(8)));

channel channel_scal   conv_ch_write    __attribute__((io("convCh")))   __attribute__((depth(CHN_DEPTH)));
channel channel_scal   conv_ch_read     __attribute__((io("convCh")))   __attribute__((depth(CHN_DEPTH)));

channel channel_scal   pool_ch_write    __attribute__((io("poolCh")))   __attribute__((depth(CHN_DEPTH)));
channel channel_scal   pool_ch_read     __attribute__((io("poolCh")))   __attribute__((depth(CHN_DEPTH)));

channel channel_scal   bypass_ch_write  __attribute__((io("bypassCh"))) __attribute__((depth(CHN_DEPTH)));
channel channel_scal   bypass_ch_read   __attribute__((io("bypassCh"))) __attribute__((depth(CHN_DEPTH)));

channel lane_data      ser_ch_write     __attribute__((io("serdeserCh")))   __attribute__((depth(0)));

channel lane_data      deser_ch_read    __attribute__((io("serdeserCh")))   __attribute__((depth(0)));
#endif


// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(lane_data input, lane_data weights)
{

	MACTYPE output = MASK_MULT & CZERO;		

	#pragma unroll	
	for (int i = 0; i < VEC_SIZE; i++) {
		output += MASK_MULT & (input.data[i] * weights.data[i]);
	}

	return output;

/*
	MACTYPE output = MASK_MULT & CZERO;
	
	#pragma unroll
	for(int i=0; i<VEC_SIZE/4; i++){
		output += MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4], input.data[i*4+1], weights.data[i*4+1], input.data[i*4+2], weights.data[i*4+2], input.data[i*4+3], weights.data[i*4+3]);
	}
	return output;
*/
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

// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memReadData(
			// Params Ports
			uchar  data_dim1,
			uchar  data_dim2,
			ushort data_dim1xdim2,
			uchar  weight_dim1,
			uchar  weight_dim2,
			ushort weight_dim3,
			ushort weight_dim4_div_lane, // avoid generating divider
			uchar  weight_dim1x2,
			uint   weight_dim1x2x3,
			uchar  conv_x,
			//uchar  conv_y,           // not used in this version
			uchar  stride,
			uchar  padding,
			uchar  split,
			uchar  group_num_x,
			uchar  group_num_y,
			uchar  group_rem_size_x,
			//uchar  group_rem_size_y, // not used in this version
			uint   group_rem_size_xyz,
			uchar  win_size_x,
			uchar  win_size_y,
			uint   win_size_xyz,  
			// Data Ports
			__global lane_data    *restrict bottom,
			__global channel_vec  *restrict weights,
			__global channel_scal *restrict bias        )

{


	// Input Data, Weights and Bias
	lane_data     data_vec;
	channel_vec   data_ch_vec;
		
	// virtual loop counters
	ushort gp_num_x, gp_num_y, out_idx_z;
	ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;
	uchar  win_itm_x, win_itm_y;
	ushort win_itm_z;
	uchar flag; // ping-pong flag

	// Ping-pong buffer
	 __local lane_data    win_buffer[2][WIN_BUF_SIZE]; // working sequence 0->1->0->1 ...
	// Weight buffer
	//__local channel_vec  weight_buffer[WEIGHT_BUF_SIZE];

	
	// Initialize the winbuf with the data in the first iteration of the group looping (as gp_num_x_winbuf=0, gp_num_y_winbuf=0)

	
	for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
		for(unsigned char  win_itm_y=0; win_itm_y<win_size_y; win_itm_y++){
			for(unsigned char  win_itm_x=0; win_itm_x<win_size_x; win_itm_x++){
				ushort feature_idx_dim1, feature_idx_dim2;
				ushort feature_idx_dim3;
				lane_data data_vec;

				feature_idx_dim1 = win_itm_x;
				feature_idx_dim2 = win_itm_y;
				feature_idx_dim3 = win_itm_z;
	
				if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
			
					data_vec = bottom[feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
				}
				else{
					#pragma unroll
					for(unsigned char vv=0; vv<VEC_SIZE; vv++){
						data_vec.data[vv] = CZERO;
					}
				}
			
				win_buffer[0][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;
			}
		}
	}

	

	if(group_num_x==1)
		gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
	else
		gp_num_x_winbuf = 1; // loop start from the second group
	gp_num_y_winbuf = 0;
	out_idx_z_winbuf = 0;
	
	// reset global group virtual loop counters
	gp_num_x = 0;
	gp_num_y = 0;
	out_idx_z = 0;

	// #pragma ivdep array(win_buffer)
	for(unsigned int out_idx_xyz=0; out_idx_xyz<(weight_dim4_div_lane*group_num_y*group_num_x); out_idx_xyz++){
		ushort        data_offset = 0; // assuming the 1st layer is not in split
		
		uchar  	      output_idx_dim1, output_idx_dim2;
		ushort	      output_idx_dim3;
		uchar  	      gp_item_idx_x;
		uchar  	      win_itm_x, win_itm_y;
		ushort 	      win_itm_z;
		uint   item_loop_bound;

		// special case when split==1, the output feature maps depend on only half the input feature maps
		if(split==0)
			data_offset = 0;
		else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
			data_offset = 0;
		else
			data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input
	
		flag = out_idx_xyz & 0x01; //ping-pong flag
				
		// reset output loop counters
		output_idx_dim1 = 0;
		output_idx_dim2 = 0;
		output_idx_dim3 = 0;
		// reset in-group item counters 
		gp_item_idx_x = 0;
				
		// reset input winbuffer loop counters
		win_itm_x = 0;
		win_itm_y = 0;
		win_itm_z = 0;
				
				
		if(gp_num_x==group_num_x-1)
			item_loop_bound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);
		else{
			item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
		}

		#pragma ivdep array(win_buffer)
		for(unsigned int win_itm_xyz = 0; win_itm_xyz < item_loop_bound; win_itm_xyz++) {
			ushort feature_idx_dim1, feature_idx_dim2;
			ushort feature_idx_dim3;
			// Winbuffer loading operations
			if(win_itm_z<weight_dim3/VEC_SIZE){
				
				lane_data data_vec;
				feature_idx_dim1 = win_itm_x+gp_num_x_winbuf*CONV_GP_SIZE_X*stride;
				feature_idx_dim2 = win_itm_y+gp_num_y_winbuf*CONV_GP_SIZE_Y*stride;
				feature_idx_dim3 = win_itm_z;

				if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
								
					data_vec = bottom[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
				}
				else{ // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
					#pragma unroll
					for(unsigned char vv=0; vv<VEC_SIZE; vv++){
						data_vec.data[vv] = CZERO;
					}
				}
				win_buffer[(~flag)&0x01][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;	

				// used as loop counters
				if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
					win_itm_z = 0;
				else if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
					win_itm_z++;

				if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
					win_itm_y = 0;
				else if(win_itm_x==win_size_x-1)
					win_itm_y++;
								
				if(win_itm_x==win_size_x-1)
					win_itm_x = 0;
				else
					win_itm_x++;
									
			}
							
			// Load weight into weight buffer
							
			// In this version, grouping is only performed in row (x) direction
			if(gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x<conv_x){
				lane_data data_vec;
				channel_vec   data_ch_vec;
				// data
				data_vec = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];

				//#pragma unroll
				//for(unsigned char ll=0; ll<LANE_NUM; ll++){
				//	data_ch_vec.lane[ll] = data_vec;
				//}
				write_channel_intel(data_ch, data_vec);



			}
			// used as output loop counters
			if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)){
				output_idx_dim3 = 0;
				gp_item_idx_x++;
			}
			else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
				output_idx_dim3++;
							
			if((output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1))
				output_idx_dim2 = 0;
			else if(output_idx_dim1==weight_dim1-1)
				output_idx_dim2++;
	                
			if(output_idx_dim1==weight_dim1-1)
				output_idx_dim1 = 0;
			else
				output_idx_dim1++;

		}

		// used as virtual group loop counters for winbuf loading operations
		if((out_idx_z_winbuf==weight_dim4_div_lane-1) && (gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
			out_idx_z_winbuf = 0;
		else if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
			out_idx_z_winbuf++;	

		if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
			gp_num_y_winbuf = 0;
		else if(gp_num_x_winbuf==group_num_x-1)
			gp_num_y_winbuf++;	

		if(gp_num_x_winbuf==group_num_x-1)
			gp_num_x_winbuf = 0;
		else
			gp_num_x_winbuf++;
		
		// used as virtual group loop counters
		if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
			out_idx_z = 0;
		else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
			out_idx_z++;	
        
		if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
			gp_num_y = 0;
		else if(gp_num_x==group_num_x-1)
			gp_num_y++;
        
		if(gp_num_x==group_num_x-1)
			gp_num_x = 0;
		else
			gp_num_x++;

	}
	
	//printf("Kernel 0 lanched !!!\n");
}

__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memReadBias(
			// Params ports
			ushort	weight_dim4_div_lane,	// avoid generating divider
			uchar 	conv_x,
			uchar	conv_y,
			__global channel_scal	*restrict bias)
{	
		

	channel_scal	bias_ch_in;
	ushort conv_xy = conv_x * conv_y;

	for (ushort i = 0; i < weight_dim4_div_lane; i++) {
		bias_ch_in = bias[i];
		for (ushort j = 0; j < conv_xy; j++) {
			write_channel_intel(bias_ch, bias_ch_in);
		}
	}


}

__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memReadWeight(
			// Params ports
			ushort	weight_dim4_div_lane,
			uchar	conv_x,
			uchar 	conv_y,
			uint 	weight_dim1x2x3,
			__global channel_vec	*restrict weights)

{

	uint conv_xxy = conv_x * conv_y;

	#pragma max_concurrency 1
	for (ushort i = 0; i < weight_dim4_div_lane; i++) {
		channel_vec weight_buffer[WEIGHT_BUF_SIZE/8];
		for (uint p = 0; p < weight_dim1x2x3/VEC_SIZE; p+=4) {
			weight_buffer[p  ] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p  ];
			weight_buffer[p+1] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+1];	
			weight_buffer[p+2] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+2];
			weight_buffer[p+3] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+3];
			// weight_buffer[p+4] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+4];
			// weight_buffer[p+5] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+5];
			// weight_buffer[p+6] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+6];
			// weight_buffer[p+7] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+7];
			// weight_buffer[p+8] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+8];
		
		}
		for (uint j = 0; j < conv_xxy; j++) {
			// channel_vec weight_buffer[WEIGHT_BUF_SIZE];
			for (uint p = 0; p < weight_dim1x2x3/VEC_SIZE; p++) {
				channel_vec weight_ch_vec;
				// if ((i & j) == 0) {
				// 	weight_buffer[p] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p];
				// }
				weight_ch_vec = weight_buffer[p];
				write_channel_intel(weight_ch, weight_ch_vec);
			}
		}
	}	 

}

__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void coreConv(
			// Params Ports
			uint  output_num,
			uint  conv_loop_cnt,
			uint  contol, //[0]-> relu  [1]->bypass pooling
			char  frac_w,
			char  frac_din,
			char  frac_dout
			)
{
	lane_data   mac_data;
 	channel_vec mac_weight;
	channel_scal bias_ch_out;
	channel_scal conv_ch_in;
	DPTYPE  bias[LANE_NUM];
	MACTYPE conv_out[LANE_NUM];
	MACTYPE lane_accum[LANE_NUM];
	MACTYPE accum_piped[LANE_NUM][PIPE_DEPTH];
	MACTYPE conv_sign_exten[LANE_NUM];
	MACTYPE conv_with_rnd_bit[LANE_NUM];
	MACTYPE conv_sum_bias[LANE_NUM];
	DPTYPE  conv_final[LANE_NUM];

	// each iteration generates one output
	for(unsigned int k=0; k<output_num; k++){
#ifndef EMULATE	
		bias_ch_out = read_channel_intel(bias_ch);
#else
		bias_ch_out = read_channel_intel(bias_ch_read);
#endif

		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			conv_out[ll] = CZERO;
			bias[ll] = bias_ch_out.lane[ll];

			#pragma unroll
			for(unsigned int p=0; p<PIPE_DEPTH; p++){
				accum_piped[ll][p] = MASK_ACCUM & CZERO;
			}
		}

		for(int j=0; j<conv_loop_cnt; j++){
#ifndef EMULATE
			mac_data = read_channel_intel(data_ch);
#else
			mac_data = read_channel_intel(data_ch_read);
#endif
#ifndef EMULATE
			mac_weight = read_channel_intel(weight_ch);
#else
			mac_weight = read_channel_intel(weight_ch_read);
#endif

			#pragma unroll
			for(unsigned char ll=0; ll<LANE_NUM; ll++){
				
				lane_accum[ll] = (MASK_ACCUM & accum_piped[ll][PIPE_DEPTH-1]) + (MASK_MULT & mac(mac_data, mac_weight.lane[ll]));
			
				#pragma unroll
				for(unsigned int p=PIPE_DEPTH-1; p>0; p-- ){
					accum_piped[ll][p] = MASK_ACCUM & accum_piped[ll][p-1];
				}
				
				accum_piped[ll][0] = MASK_ACCUM & lane_accum[ll];

				#ifdef DEBUG_CONV
				//if(ll==0 && k==0){
				//	printf("dot_cnt=%d data=%f weight=%f (loop=%d, lane= %d, vec=0)\n", k, (float)mac_data.lane[ll].data[0], (float)mac_weight.lane[ll].data[0], j, ll);
				//}
				#endif
			}
		}// end of conv loop

		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			#pragma unroll
			for(unsigned i=0; i<PIPE_DEPTH; i++){
				conv_out[ll] += accum_piped[ll][i];
			}
			
			if(conv_out[ll]>=0)
				conv_sign_exten[ll] = 0x00;
			else
				conv_sign_exten[ll] = ~(0xFFFFFFFF>>(frac_w+frac_din-frac_dout-1));
			
			conv_with_rnd_bit[ll] = (conv_sign_exten[ll] | (conv_out[ll]>>(frac_w+frac_din-frac_dout-1))) + 0x01;

			if(conv_with_rnd_bit[ll]>=256)
				conv_sum_bias[ll] = MASK9B & 0xFF;
			else if(conv_with_rnd_bit[ll]<-256)
				conv_sum_bias[ll] = MASK9B & 0x100;
			else
				conv_sum_bias[ll] = (MASK9B & conv_with_rnd_bit[ll])+(bias[ll]>>(frac_w-frac_dout-1))+0x01;

			conv_final[ll] = MASK8B & (conv_sum_bias[ll]>>0x01);
			
			// Relu operation
			if((contol&0x01)==0x01){
				if((conv_final[ll]&MASKSIGN)==MASKSIGN)
					conv_ch_in.lane[ll] = 0;
				else
					conv_ch_in.lane[ll] = conv_final[ll];
			}
			else
				conv_ch_in.lane[ll] = conv_final[ll];
			
			#ifdef DEBUG_CONV
			if(ll==0 && k==0)
				printf("dot_cnt=%d sum=%f rnd=%f sum_bias=%f final=%f (bias=%f)\n\n", k, (float)conv_out[ll], (float)conv_with_rnd_bit[ll], (float)conv_sum_bias[ll], (float)conv_final[ll], (float)bias[ll]);
			#endif

		}

		// write convoluation results
		//if((contol&0x02)==0x02)
			//by-pass pooling
#ifndef EMULATE
			write_channel_intel(bypass_ch, conv_ch_in);
#else
			write_channel_intel(bypass_ch_write, conv_ch_in);
#endif
		/*
		else // to pooling kernel
#ifndef EMULATE
			write_channel_intel(conv_ch, conv_ch_in);
#else
			write_channel_intel(conv_ch_write, conv_ch_in);
#endif
		*/
			//printf("Write channel item-%d is written in channel %d...\n", k, ll);

	}// end of output loop
 
}

/*
__kernel
__attribute__((task))
void maxPool(
			// Params Ports
			uint  input_num,
			uchar line_size,  // line_size should be no larger than POOL_LBUF_DEPTH
			uchar pool_size,  // by now, only pooling size no larger than 3
			uchar pool_stride
			
			)
{
	channel_scal conv_ch_out;
	channel_scal pool_final;

	//DPTYPE line_buf_0[LANE_NUM][POOL_LBUF_DEPTH];
	//DPTYPE line_buf_1[LANE_NUM][POOL_LBUF_DEPTH];
	DPTYPE line_buf_0[POOL_LBUF_DEPTH][LANE_NUM];
	DPTYPE line_buf_1[POOL_LBUF_DEPTH][LANE_NUM];

	uchar  line_buf_ptr;
	uchar  col_pool_cnt;
	uchar  row_pool_cnt;
	uchar  row_cnt;
	DPTYPE row_pool_reg[LANE_NUM];
	DPTYPE col_pool_reg[LANE_NUM];
	//DPTYPE pool_reg[LANE_NUM][POOL_MAX_SIZE];
	DPTYPE pool_reg[POOL_MAX_SIZE][LANE_NUM];
	
	// Each iteration consumes one output from convolution kernel
	// and then Pooling is performed in column and row directions
	line_buf_ptr = 0;
	row_pool_cnt = 0;
	col_pool_cnt = 0;
	for(unsigned int k=0; k<input_num; k++){

#ifndef EMULATE
		conv_ch_out = read_channel_intel(conv_ch);
#else
		conv_ch_out = read_channel_intel(conv_ch_read);
#endif	
		// Two line buffer to form the 3x3 pooling window
		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			if(pool_size==3)
				row_pool_reg[ll] = pool_max(line_buf_1[line_buf_ptr][ll], line_buf_0[line_buf_ptr][ll]);
			else // pool_size==2
				row_pool_reg[ll] = line_buf_0[line_buf_ptr][ll];
			
			pool_reg[0][ll] = pool_max(row_pool_reg[ll], conv_ch_out.lane[ll]);
			
			if(pool_size==3)
				col_pool_reg[ll] = pool_max(pool_reg[1][ll], pool_reg[2][ll]);
			else //pool_size==2
				col_pool_reg[ll] = pool_reg[1][ll];

			pool_final.lane[ll] = pool_max(col_pool_reg[ll], pool_reg[0][ll]);

			line_buf_1[line_buf_ptr][ll] = line_buf_0[line_buf_ptr][ll];
			line_buf_0[line_buf_ptr][ll] = conv_ch_out.lane[ll];

			#pragma unroll
			for(unsigned char p=POOL_MAX_SIZE-1; p>0; p--){
				pool_reg[p][ll]=pool_reg[p-1][ll];
			}
		}
		
		#ifdef DEBUG_POOL
		printf("Maxpool input_num=%d, line_buf_ptr=%d, row_pool_cnt=%d, col_pool_cnt=%d\n", k, line_buf_ptr, row_pool_cnt, col_pool_cnt);
		printf("        row_cnt=%d\n", row_cnt);
		#endif
		
		// Generates pooling pipeline register wr/rd pointer
		if(row_pool_cnt==(pool_size-1)){

			if(col_pool_cnt==(pool_size-1)){
#ifndef EMULATE
				write_channel_intel(pool_ch, pool_final);
#else
				write_channel_intel(pool_ch_write, pool_final);
#endif
				#ifdef DEBUG_POOL
				printf("        reg0=%f, reg1=%f, reg2=%f, max=%f\n", (float)pool_reg[0][0], (float)pool_reg[1][0], (float)pool_reg[2][0], (float)pool_final.lane[0]);
				#endif

				col_pool_cnt = (pool_size-pool_stride);
			}
			else
				col_pool_cnt = col_pool_cnt + 1;
		}
		else
			col_pool_cnt = 0;

		// Generates line buffer wr/rd pointer
		if(line_buf_ptr==(line_size-1)){
			line_buf_ptr = 0;

			// Row counters for recognize frames
			if(row_cnt == (line_size-1)) // assuming row_num = line_size, i.e. rectangular frame
				row_cnt = 0;
			else
				row_cnt = row_cnt + 1;

			// Pooling window slide counter for rows
			if(row_cnt == 0)
				row_pool_cnt = 0;
			else if(row_pool_cnt==(pool_size-1))
				row_pool_cnt = (pool_size-pool_stride);
			else
				row_pool_cnt = row_pool_cnt + 1;
		}
		else{
			line_buf_ptr = line_buf_ptr + 1;
		}

	}
}
*/

// Store Data to Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
				// Params Ports
				uchar  out_dim1,
				uchar  out_dim2,
				ushort out_dim3,
				ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
				uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
				uchar  batch_indx_dim1,
				uchar  batch_indx_dim2,
				uchar  bypass,
				uchar  padd_offset,
				// Data Ports
                __global DPTYPE *restrict top
				)
{
	uchar  global_x = get_global_id(0); // max value 256
	uchar  global_y = get_global_id(1); // max value 256
	ushort global_z = get_global_id(2); // max value 4096
	uchar  local_x 	= get_local_id(0); // max value 256
	uchar  local_y 	= get_local_id(1); // max value 256
	uchar  local_z 	= get_local_id(2); // max value 256

	uchar  index_z_item; // max value 256
	ushort index_z_group;// max value 4096

	channel_scal   output;
	__local DPTYPE buffer[LANE_NUM];

	if(local_z==0){
//		if((bypass&0x01)==0x01)
#ifndef EMULATE
			output = read_channel_intel(bypass_ch);
#else
			output = read_channel_intel(bypass_ch_read);
#endif
//		else
/*
#ifndef EMULATE
			output = read_channel_intel(pool_ch);
#else
			output = read_channel_intel(pool_ch_read);
#endif
*/
		#pragma unroll
		for(uchar ll=0; ll<LANE_NUM; ll++){
			buffer[ll]=output.lane[ll];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	// fetch data from local buffer and write back to DDR
	index_z_group = (global_z-padd_offset)/VEC_SIZE;
	index_z_item  = (global_z-padd_offset)%VEC_SIZE;

	if((global_z-padd_offset)<out_dim3 && (global_z>=padd_offset)){

		top[index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];

		#ifdef DEBUG_MEMWR
		//if((global_z-padd_offset) == 0){
			//for(unsigned char ll=0; ll<LANE_NUM; ll++){
			printf("MemWr results= %f (x=%d, y=%d, z=%d, ll=%d)\n", (float)output.lane[0], global_x, global_y, global_z, 0);
			//}
		//	}
		#endif

	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

}

/*
__kernel
__attribute__((max_work_group_size(1,1,LRN_MAX_LOCAL_SIZE)))
void lrn(
			// Params Ports
			uchar data_dim1,
			uchar data_dim2,
			char  frac_dout,
			// Data Ports
			__global lane_data *restrict bottom,
			__global lane_data *restrict top
		)
{
	uchar  global_x = get_global_id(0); // max value 256
	uchar  global_y = get_global_id(1); // max value 256
	ushort global_z = get_global_id(2); // max value 4096

	#ifdef DEBUG_LRN
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_z = get_local_id(2);
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);
	#endif
	
	__local DPTYPE z_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE+LRN_WIN_SIZE]; // allocate two more points for padding
	__local DPTYPE lrn_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE];
	channel_scal data_in;
	channel_scal data_pad_left;
	channel_scal data_pad_right;
	channel_scal data_out;
	lane_data    data_in_partial;
	lane_data    data_left_partial;
	lane_data    data_right_partial;
	lane_data    data_out_partial;
	int          *convert_ptr;
	int          expo;
	uint         manti;
	uint         addr_1, addr_2, addr;
	float        lrn_reg1, lrn_reg2, lrn_tmp, lrn_out;
	short        lrn_cnvt, lrn_cnvt2;
	
	// Load the all data in one line along dim3 into local line buffer
	#pragma unroll
	for(unsigned char ll=0; ll<VEC_SIZE; ll++){
		z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2] = bottom[global_z*data_dim2*data_dim1 + global_y*data_dim1+ global_x].data[ll];
	}
	
	//Padding left
	if(global_z==0){
		#pragma unroll
		for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++){
			z_buffer[ll] = CZERO;
		}
	}

	// Padding right
	if(global_z==(get_global_size(2)-1)){
		#pragma unroll
		for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++){
			z_buffer[VEC_SIZE*get_local_size(2)+ll+LRN_WIN_SIZE/2] = CZERO;
		}
	}

	#ifdef DEBUG_LRN
	if(global_z==0&&global_x==0&&global_y==0)
	printf("Kernel LRN: work-item x=%d, y=%d, z=%d(z_local=%d)\n", global_x, global_y, global_z, local_z);
	#endif
	barrier(CLK_LOCAL_MEM_FENCE);

	// Piecewise interpolation pipeline for lrn operation (i.e., y=pwlf(x'))
	for(unsigned char ll=0; ll<VEC_SIZE; ll++){
		// First Step: Coefficients table looking-up
		// Calculate x'=sum(x(k)^2) for the pwlf function, x(k)s are from adjacent featuremaps
		lrn_reg2 = CZERO;
		#pragma unroll
		for(char k=-LRN_WIN_SIZE/2; k<=LRN_WIN_SIZE/2; k++){
			lrn_cnvt = z_buffer[global_z*VEC_SIZE+ll+k+LRN_WIN_SIZE/2]<<(-frac_dout);
			lrn_reg1 = convert_float(lrn_cnvt);
			lrn_reg2 += lrn_reg1 * lrn_reg1;
			#ifdef DEBUG_LRN
			if(global_z==0&&global_x==0&&global_y==0)
			printf("x=%f(k=%d), ", lrn_reg1, k);
			#endif
		}
		convert_ptr = (int*) (&lrn_reg2);
		expo = (EXP_MASK & (*convert_ptr >> MAN_BITS)) - 127;
		manti = ((*convert_ptr) & MAN_MASK);
		
		addr_1 = ((expo-EXP_STEP_MIN)>>EXP_STEP_LOG)<<MAN_INDEX_BITS;
		addr_2 = (manti>>(MAN_BITS-MAN_INDEX_BITS) & MAN_INDEX_MASK)+1;
		if(expo<EXP_STEP_MIN)
			addr = 0;
		else
			addr = addr_1+addr_2;

		lrn_tmp = ((lrn_reg2-x_sample[addr])*h_inv[addr])*coef1[addr] + coef0[addr];	
		
		lrn_cnvt2 = z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2]<<(-frac_dout);
		lrn_out = lrn_tmp*convert_float(lrn_cnvt2);

		// Convert float to DPTYPE fixed-point
		// Note: current version only support frac_din=0 for next layer
		lrn_buffer[global_z*VEC_SIZE+ll] = convert_char_rte(lrn_out);

		#ifdef DEBUG_LRN
		if(global_z==0&&global_x==0&&global_y==0)
		printf("\nKernel LRN (ll=%d): pwlf_x=%f, expo=%d, addr=%d, pwlf_y=%f, lrn=%f\n", ll, lrn_reg2, expo, addr, lrn_tmp, lrn_out);
		#endif
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the results back to global mem
	#pragma unroll
	for(unsigned char vv=0; vv<VEC_SIZE; vv++){
		data_out_partial.data[vv]=lrn_buffer[global_z*VEC_SIZE+vv];
	}
	top[global_z*data_dim2*data_dim1 + global_y*data_dim1 + global_x] = data_out_partial;

	if (global_z >= 0 && global_z <= 3) {
		if (global_y >= 0 && global_y <= 3) {
			if (global_x >= 0 && global_x <= 3) {
				lane_data temp;

				#pragma unroll
				for (unsigned char ll = 0; ll < VEC_SIZE; ll++) {
					temp.data[ll] = top[global_z*data_dim2*data_dim1 + global_y*data_dim1 * global_x].data[ll];
				}
			}
		}
	}
	
	#ifdef DEBUG_LRN_OUT
	if(global_z==0&&global_x==0&&global_y==0)
	printf("\nKernel LRN OUT: x=%d, y=%d, z=%d, result=%f\n", global_x, global_y, global_z, (float)data_out_partial.data[0]);
	#endif

}

*/

#ifdef EMULATE

__kernel
__attribute__((task))
void lrnSer(
		// Param Ports
		uchar  data_dim1,
		uchar  data_dim2,
		ushort data_dim3,
		// Data Ports
		__global lane_data *restrict bottom
		)

{	

	for (unsigned short dim3 = 0; dim3 < data_dim3; dim3++) {
		for (unsigned char dim2 = 0; dim2 < data_dim2; dim2++) {
			for (unsigned char dim1 = 0; dim1 < data_dim1; dim1++) {
				lane_data buf;

				#pragma unroll
				for (unsigned char ll = 0; ll < VEC_SIZE; ll++)
					buf.data[ll] = bottom[dim3*data_dim2*data_dim1 + dim2*data_dim1 + dim1].data[ll];

				write_channel_intel (ser_ch_write, buf);
			}
		}
	}

}

__kernel
__attribute__((task))
void memReadDeser(
	// Param Ports,
	uchar  data_dim1,
	uchar  data_dim2,
	ushort data_dim3,
	// Data Ports
	__global lane_data *restrict top
	)

{


	for (unsigned short dim3 = 0; dim3 < data_dim3; dim3++) {
		for (unsigned char dim2 = 0; dim2 < data_dim2; dim2++) {
			for (unsigned char dim1 = 0; dim1 < data_dim1; dim1++) {
				lane_data buf;

				buf = read_channel_intel (deser_ch_read);

				#pragma unroll
				for (unsigned char ll = 0; ll < VEC_SIZE; ll++)
					top[dim3*data_dim2*data_dim1 + dim2*data_dim1 + dim1].data[ll] = buf.data[ll];

			}
		}
	}

}

#endif

