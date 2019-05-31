// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE0() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[0]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[1], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;
		
		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[0]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[0]);
			}

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[0]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[1], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				/*

				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/
				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on

				channel_cols0 toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels0, toNext);
			}

			out_ch++;

		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE1() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[1]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[2], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[1]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[1]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[1]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[2], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols0 fromPrev;
				channel_cols1 toNext;
				fromPrev = read_channel_intel(chain_output_channels0);
				#pragma unroll
				for (int col = 0; col < 1; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[1] = accumulation;
				write_channel_intel(chain_output_channels1, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE2() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[2]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[3], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[2]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[2]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[2]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[3], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols1 fromPrev;
				channel_cols2 toNext;
				fromPrev = read_channel_intel(chain_output_channels1);
				#pragma unroll
				for (int col = 0; col < 2; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[2] = accumulation;
				write_channel_intel(chain_output_channels2, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE3() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[3]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[4], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[3]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[3]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[3]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[4], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols2 fromPrev;
				channel_cols3 toNext;
				fromPrev = read_channel_intel(chain_output_channels2);
				#pragma unroll
				for (int col = 0; col < 3; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[3] = accumulation;
				write_channel_intel(chain_output_channels3, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE4() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[4]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[5], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[4]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[4]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[4]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[5], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols3 fromPrev;
				channel_cols4 toNext;
				fromPrev = read_channel_intel(chain_output_channels3);
				#pragma unroll
				for (int col = 0; col < 4; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[4] = accumulation;
				write_channel_intel(chain_output_channels4, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE5() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[5]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[6], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[5]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[5]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[5]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[6], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols4 fromPrev;
				channel_cols5 toNext;
				fromPrev = read_channel_intel(chain_output_channels4);
				#pragma unroll
				for (int col = 0; col < 5; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[5] = accumulation;
				write_channel_intel(chain_output_channels5, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE6() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[6]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[7], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[6]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[6]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[6]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[7], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols5 fromPrev;
				channel_cols6 toNext;
				fromPrev = read_channel_intel(chain_output_channels5);
				#pragma unroll
				for (int col = 0; col < 6; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[6] = accumulation;
				write_channel_intel(chain_output_channels6, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE7() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[7]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[8], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[7]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[7]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[7]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[8], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols6 fromPrev;
				channel_cols7 toNext;
				fromPrev = read_channel_intel(chain_output_channels6);
				#pragma unroll
				for (int col = 0; col < 7; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[7] = accumulation;
				write_channel_intel(chain_output_channels7, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE8() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[8]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[9], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[8]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[8]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[8]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[9], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols7 fromPrev;
				channel_cols8 toNext;
				fromPrev = read_channel_intel(chain_output_channels7);
				#pragma unroll
				for (int col = 0; col < 8; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[8] = accumulation;
				write_channel_intel(chain_output_channels8, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE9() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[9]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[10], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[9]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[9]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[9]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[10], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols8 fromPrev;
				channel_cols9 toNext;
				fromPrev = read_channel_intel(chain_output_channels8);
				#pragma unroll
				for (int col = 0; col < 9; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[9] = accumulation;
				write_channel_intel(chain_output_channels9, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE10() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[10]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[11], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[10]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[10]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[10]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[11], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols9 fromPrev;
				channel_cols10 toNext;
				fromPrev = read_channel_intel(chain_output_channels9);
				#pragma unroll
				for (int col = 0; col < 10; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[10] = accumulation;
				write_channel_intel(chain_output_channels10, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE11() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[11]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[12], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[11]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[11]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[11]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[12], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols10 fromPrev;
				channel_cols11 toNext;
				fromPrev = read_channel_intel(chain_output_channels10);
				#pragma unroll
				for (int col = 0; col < 11; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[11] = accumulation;
				write_channel_intel(chain_output_channels11, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE12() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[12]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[13], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[12]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[12]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[12]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[13], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols11 fromPrev;
				channel_cols12 toNext;
				fromPrev = read_channel_intel(chain_output_channels11);
				#pragma unroll
				for (int col = 0; col < 12; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[12] = accumulation;
				write_channel_intel(chain_output_channels12, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE13() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[13]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[14], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[13]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[13]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[13]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[14], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols12 fromPrev;
				channel_cols13 toNext;
				fromPrev = read_channel_intel(chain_output_channels12);
				#pragma unroll
				for (int col = 0; col < 13; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[13] = accumulation;
				write_channel_intel(chain_output_channels13, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE14() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[14]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[15], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[14]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[14]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[14]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[15], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols13 fromPrev;
				channel_cols14 toNext;
				fromPrev = read_channel_intel(chain_output_channels13);
				#pragma unroll
				for (int col = 0; col < 14; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[14] = accumulation;
				write_channel_intel(chain_output_channels14, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE15() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[15]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[16], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[15]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[15]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[15]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[16], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols14 fromPrev;
				channel_cols15 toNext;
				fromPrev = read_channel_intel(chain_output_channels14);
				#pragma unroll
				for (int col = 0; col < 15; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[15] = accumulation;
				write_channel_intel(chain_output_channels15, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE16() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[16]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[17], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[16]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[16]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[16]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[17], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols15 fromPrev;
				channel_cols16 toNext;
				fromPrev = read_channel_intel(chain_output_channels15);
				#pragma unroll
				for (int col = 0; col < 16; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[16] = accumulation;
				write_channel_intel(chain_output_channels16, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE17() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[17]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[18], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[17]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[17]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[17]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[18], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols16 fromPrev;
				channel_cols17 toNext;
				fromPrev = read_channel_intel(chain_output_channels16);
				#pragma unroll
				for (int col = 0; col < 17; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[17] = accumulation;
				write_channel_intel(chain_output_channels17, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE18() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[18]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[19], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[18]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[18]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[18]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[19], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols17 fromPrev;
				channel_cols18 toNext;
				fromPrev = read_channel_intel(chain_output_channels17);
				#pragma unroll
				for (int col = 0; col < 18; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[18] = accumulation;
				write_channel_intel(chain_output_channels18, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE19() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[19]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[20], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[19]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[19]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[19]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[20], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols18 fromPrev;
				channel_cols19 toNext;
				fromPrev = read_channel_intel(chain_output_channels18);
				#pragma unroll
				for (int col = 0; col < 19; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[19] = accumulation;
				write_channel_intel(chain_output_channels19, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE20() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[20]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[21], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[20]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[20]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[20]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[21], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols19 fromPrev;
				channel_cols20 toNext;
				fromPrev = read_channel_intel(chain_output_channels19);
				#pragma unroll
				for (int col = 0; col < 20; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[20] = accumulation;
				write_channel_intel(chain_output_channels20, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE21() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[21]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[22], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[21]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[21]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[21]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[22], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols20 fromPrev;
				channel_cols21 toNext;
				fromPrev = read_channel_intel(chain_output_channels20);
				#pragma unroll
				for (int col = 0; col < 21; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[21] = accumulation;
				write_channel_intel(chain_output_channels21, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE22() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[22]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[23], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[22]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[22]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[22]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[23], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols21 fromPrev;
				channel_cols22 toNext;
				fromPrev = read_channel_intel(chain_output_channels21);
				#pragma unroll
				for (int col = 0; col < 22; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[22] = accumulation;
				write_channel_intel(chain_output_channels22, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE23() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[23]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[24], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[23]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[23]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[23]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[24], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols22 fromPrev;
				channel_cols23 toNext;
				fromPrev = read_channel_intel(chain_output_channels22);
				#pragma unroll
				for (int col = 0; col < 23; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[23] = accumulation;
				write_channel_intel(chain_output_channels23, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE24() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[24]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[25], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[24]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[24]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[24]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[25], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols23 fromPrev;
				channel_cols24 toNext;
				fromPrev = read_channel_intel(chain_output_channels23);
				#pragma unroll
				for (int col = 0; col < 24; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[24] = accumulation;
				write_channel_intel(chain_output_channels24, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE25() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[25]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[26], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[25]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[25]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[25]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[26], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols24 fromPrev;
				channel_cols25 toNext;
				fromPrev = read_channel_intel(chain_output_channels24);
				#pragma unroll
				for (int col = 0; col < 25; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[25] = accumulation;
				write_channel_intel(chain_output_channels25, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE26() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[26]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[27], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[26]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[26]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[26]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[27], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols25 fromPrev;
				channel_cols26 toNext;
				fromPrev = read_channel_intel(chain_output_channels25);
				#pragma unroll
				for (int col = 0; col < 26; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[26] = accumulation;
				write_channel_intel(chain_output_channels26, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE27() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[27]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[28], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[27]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[27]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[27]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[28], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols26 fromPrev;
				channel_cols27 toNext;
				fromPrev = read_channel_intel(chain_output_channels26);
				#pragma unroll
				for (int col = 0; col < 27; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[27] = accumulation;
				write_channel_intel(chain_output_channels27, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE28() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[28]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[29], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[28]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[28]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[28]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[29], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols27 fromPrev;
				channel_cols28 toNext;
				fromPrev = read_channel_intel(chain_output_channels27);
				#pragma unroll
				for (int col = 0; col < 28; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[28] = accumulation;
				write_channel_intel(chain_output_channels28, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE29() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[29]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[30], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[29]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[29]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[29]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[30], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols28 fromPrev;
				channel_cols29 toNext;
				fromPrev = read_channel_intel(chain_output_channels28);
				#pragma unroll
				for (int col = 0; col < 29; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[29] = accumulation;
				write_channel_intel(chain_output_channels29, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE30() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[30]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[31], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[30]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[30]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[30]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[31], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols29 fromPrev;
				channel_cols30 toNext;
				fromPrev = read_channel_intel(chain_output_channels29);
				#pragma unroll
				for (int col = 0; col < 30; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[30] = accumulation;
				write_channel_intel(chain_output_channels30, toNext);
			}

			out_ch++;
			
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE31() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[31]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[32], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		char num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			bias = read_channel_intel(bias_channels[31]);
			for (char i = 0; i < num_weight_plates; i++) {
				// Case 1:
				// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
				// write_channel_intel(chain_weight_channels[id+1], temp_weight);
				// weight_buffer[i] = temp_weight.weight[id];
				weight_buffer[i] = read_channel_intel(weight_channels[31]);
			}			

			
			for (char brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					lane_cols feature = read_channel_intel(chain_data_channels[31]);

					// Bypassing the data to next PE
					write_channel_intel(chain_data_channels[32], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				
				/*
				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/

				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on
				channel_cols30 fromPrev;
				channel_cols31 toNext;
				fromPrev = read_channel_intel(chain_output_channels30);
				#pragma unroll
				for (int col = 0; col < 31; col++) {
					toNext.cols[col] = fromPrev.cols[col];
				}
				toNext.cols[31] = accumulation;
				write_channel_intel(chain_output_channels31, toNext);
			}

			out_ch++;
			
		}

	}

}


