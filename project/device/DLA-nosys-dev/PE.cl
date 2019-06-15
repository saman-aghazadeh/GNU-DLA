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

	int iter = 0; 

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE0][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);
		
		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE0][%d] handling out channel %d\n", iter, out_ch);

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

			w_data accumulation;
			uint brick = 0;
			uint i = 0;	
			//for (int brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;
			
				// printf ("[FPGA][PE0][%d] Handling brick %d\n", iter, brick);	

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE0][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[0]);

					// Bypassing the data to next PE
					// printf ("[FPGA][PE0][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[1], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
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

				if (i == conv_loop_cnt-1) {
					channel_cols0 toNext;
					toNext.cols[0] = accumulation;
					write_channel_intel(chain_output_channels0, toNext);
					// printf ("[FPGA][PE0][%d] written somethng to the output channel!\n", iter); 
					i = 0;
					brick++;
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
				} else {
					i++;
				}
			}

			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE1][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE1][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE1][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE1][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[1]);

					// printf ("[FPGA][PE1][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE1][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[2], feature);

					// printf ("[FPGA][PE1][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols0 fromPrev;
					channel_cols1 toNext;
					fromPrev = read_channel_intel(chain_output_channels0);
					#pragma unroll
					for (int col = 0; col < 1; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[1] = accumulation;
					write_channel_intel(chain_output_channels1, toNext);
					// printf ("[FPGA][PE1][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE2][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE2][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE2][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE2][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[2]);

					// printf ("[FPGA][PE2][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE2][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[3], feature);

					// printf ("[FPGA][PE2][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols1 fromPrev;
					channel_cols2 toNext;
					fromPrev = read_channel_intel(chain_output_channels1);
					#pragma unroll
					for (int col = 0; col < 2; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[2] = accumulation;
					write_channel_intel(chain_output_channels2, toNext);
					// printf ("[FPGA][PE2][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE3][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE3][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE3][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE3][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[3]);

					// printf ("[FPGA][PE3][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE3][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[4], feature);

					// printf ("[FPGA][PE3][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols2 fromPrev;
					channel_cols3 toNext;
					fromPrev = read_channel_intel(chain_output_channels2);
					#pragma unroll
					for (int col = 0; col < 3; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[3] = accumulation;
					write_channel_intel(chain_output_channels3, toNext);
					// printf ("[FPGA][PE3][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE4][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE4][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE4][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE4][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[4]);

					// printf ("[FPGA][PE4][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE4][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[5], feature);

					// printf ("[FPGA][PE4][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols3 fromPrev;
					channel_cols4 toNext;
					fromPrev = read_channel_intel(chain_output_channels3);
					#pragma unroll
					for (int col = 0; col < 4; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[4] = accumulation;
					write_channel_intel(chain_output_channels4, toNext);
					// printf ("[FPGA][PE4][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE5][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE5][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE5][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE5][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[5]);

					// printf ("[FPGA][PE5][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE5][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[6], feature);

					// printf ("[FPGA][PE5][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols4 fromPrev;
					channel_cols5 toNext;
					fromPrev = read_channel_intel(chain_output_channels4);
					#pragma unroll
					for (int col = 0; col < 5; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[5] = accumulation;
					write_channel_intel(chain_output_channels5, toNext);
					// printf ("[FPGA][PE5][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE6][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE6][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE6][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE6][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[6]);

					// printf ("[FPGA][PE6][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE6][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[7], feature);

					// printf ("[FPGA][PE6][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols5 fromPrev;
					channel_cols6 toNext;
					fromPrev = read_channel_intel(chain_output_channels5);
					#pragma unroll
					for (int col = 0; col < 6; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[6] = accumulation;
					write_channel_intel(chain_output_channels6, toNext);
					// printf ("[FPGA][PE6][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE7][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE7][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE7][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE7][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[7]);

					// printf ("[FPGA][PE7][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE7][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[8], feature);

					// printf ("[FPGA][PE7][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols6 fromPrev;
					channel_cols7 toNext;
					fromPrev = read_channel_intel(chain_output_channels6);
					#pragma unroll
					for (int col = 0; col < 7; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[7] = accumulation;
					write_channel_intel(chain_output_channels7, toNext);
					// printf ("[FPGA][PE7][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE8][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE8][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE8][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE8][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[8]);

					// printf ("[FPGA][PE8][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE8][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[9], feature);

					// printf ("[FPGA][PE8][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols7 fromPrev;
					channel_cols8 toNext;
					fromPrev = read_channel_intel(chain_output_channels7);
					#pragma unroll
					for (int col = 0; col < 8; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[8] = accumulation;
					write_channel_intel(chain_output_channels8, toNext);
					// printf ("[FPGA][PE8][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE9][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE9][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE9][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE9][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[9]);

					// printf ("[FPGA][PE9][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE9][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[10], feature);

					// printf ("[FPGA][PE9][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols8 fromPrev;
					channel_cols9 toNext;
					fromPrev = read_channel_intel(chain_output_channels8);
					#pragma unroll
					for (int col = 0; col < 9; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[9] = accumulation;
					write_channel_intel(chain_output_channels9, toNext);
					// printf ("[FPGA][PE9][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE10][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE10][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE10][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE10][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[10]);

					// printf ("[FPGA][PE10][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE10][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[11], feature);

					// printf ("[FPGA][PE10][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols9 fromPrev;
					channel_cols10 toNext;
					fromPrev = read_channel_intel(chain_output_channels9);
					#pragma unroll
					for (int col = 0; col < 10; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[10] = accumulation;
					write_channel_intel(chain_output_channels10, toNext);
					// printf ("[FPGA][PE10][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE11][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE11][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE11][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE11][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[11]);

					// printf ("[FPGA][PE11][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE11][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[12], feature);

					// printf ("[FPGA][PE11][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols10 fromPrev;
					channel_cols11 toNext;
					fromPrev = read_channel_intel(chain_output_channels10);
					#pragma unroll
					for (int col = 0; col < 11; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[11] = accumulation;
					write_channel_intel(chain_output_channels11, toNext);
					// printf ("[FPGA][PE11][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE12][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE12][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE12][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE12][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[12]);

					// printf ("[FPGA][PE12][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE12][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[13], feature);

					// printf ("[FPGA][PE12][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols11 fromPrev;
					channel_cols12 toNext;
					fromPrev = read_channel_intel(chain_output_channels11);
					#pragma unroll
					for (int col = 0; col < 12; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[12] = accumulation;
					write_channel_intel(chain_output_channels12, toNext);
					// printf ("[FPGA][PE12][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE13][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE13][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE13][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE13][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[13]);

					// printf ("[FPGA][PE13][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE13][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[14], feature);

					// printf ("[FPGA][PE13][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols12 fromPrev;
					channel_cols13 toNext;
					fromPrev = read_channel_intel(chain_output_channels12);
					#pragma unroll
					for (int col = 0; col < 13; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[13] = accumulation;
					write_channel_intel(chain_output_channels13, toNext);
					// printf ("[FPGA][PE13][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE14][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE14][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE14][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE14][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[14]);

					// printf ("[FPGA][PE14][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE14][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[15], feature);

					// printf ("[FPGA][PE14][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols13 fromPrev;
					channel_cols14 toNext;
					fromPrev = read_channel_intel(chain_output_channels13);
					#pragma unroll
					for (int col = 0; col < 14; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[14] = accumulation;
					write_channel_intel(chain_output_channels14, toNext);
					// printf ("[FPGA][PE14][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE15][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE15][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE15][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE15][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[15]);

					// printf ("[FPGA][PE15][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE15][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[16], feature);

					// printf ("[FPGA][PE15][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols14 fromPrev;
					channel_cols15 toNext;
					fromPrev = read_channel_intel(chain_output_channels14);
					#pragma unroll
					for (int col = 0; col < 15; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[15] = accumulation;
					write_channel_intel(chain_output_channels15, toNext);
					// printf ("[FPGA][PE15][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE16][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE16][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE16][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE16][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[16]);

					// printf ("[FPGA][PE16][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE16][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[17], feature);

					// printf ("[FPGA][PE16][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols15 fromPrev;
					channel_cols16 toNext;
					fromPrev = read_channel_intel(chain_output_channels15);
					#pragma unroll
					for (int col = 0; col < 16; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[16] = accumulation;
					write_channel_intel(chain_output_channels16, toNext);
					// printf ("[FPGA][PE16][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE17][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE17][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE17][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE17][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[17]);

					// printf ("[FPGA][PE17][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE17][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[18], feature);

					// printf ("[FPGA][PE17][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols16 fromPrev;
					channel_cols17 toNext;
					fromPrev = read_channel_intel(chain_output_channels16);
					#pragma unroll
					for (int col = 0; col < 17; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[17] = accumulation;
					write_channel_intel(chain_output_channels17, toNext);
					// printf ("[FPGA][PE17][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE18][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE18][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE18][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE18][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[18]);

					// printf ("[FPGA][PE18][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE18][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[19], feature);

					// printf ("[FPGA][PE18][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols17 fromPrev;
					channel_cols18 toNext;
					fromPrev = read_channel_intel(chain_output_channels17);
					#pragma unroll
					for (int col = 0; col < 18; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[18] = accumulation;
					write_channel_intel(chain_output_channels18, toNext);
					// printf ("[FPGA][PE18][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE19][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE19][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE19][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE19][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[19]);

					// printf ("[FPGA][PE19][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE19][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[20], feature);

					// printf ("[FPGA][PE19][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols18 fromPrev;
					channel_cols19 toNext;
					fromPrev = read_channel_intel(chain_output_channels18);
					#pragma unroll
					for (int col = 0; col < 19; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[19] = accumulation;
					write_channel_intel(chain_output_channels19, toNext);
					// printf ("[FPGA][PE19][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE20][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE20][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE20][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE20][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[20]);

					// printf ("[FPGA][PE20][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE20][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[21], feature);

					// printf ("[FPGA][PE20][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols19 fromPrev;
					channel_cols20 toNext;
					fromPrev = read_channel_intel(chain_output_channels19);
					#pragma unroll
					for (int col = 0; col < 20; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[20] = accumulation;
					write_channel_intel(chain_output_channels20, toNext);
					// printf ("[FPGA][PE20][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE21][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE21][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE21][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE21][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[21]);

					// printf ("[FPGA][PE21][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE21][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[22], feature);

					// printf ("[FPGA][PE21][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols20 fromPrev;
					channel_cols21 toNext;
					fromPrev = read_channel_intel(chain_output_channels20);
					#pragma unroll
					for (int col = 0; col < 21; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[21] = accumulation;
					write_channel_intel(chain_output_channels21, toNext);
					// printf ("[FPGA][PE21][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE22][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE22][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE22][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE22][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[22]);

					// printf ("[FPGA][PE22][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE22][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[23], feature);

					// printf ("[FPGA][PE22][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols21 fromPrev;
					channel_cols22 toNext;
					fromPrev = read_channel_intel(chain_output_channels21);
					#pragma unroll
					for (int col = 0; col < 22; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[22] = accumulation;
					write_channel_intel(chain_output_channels22, toNext);
					// printf ("[FPGA][PE22][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE23][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE23][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE23][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE23][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[23]);

					// printf ("[FPGA][PE23][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE23][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[24], feature);

					// printf ("[FPGA][PE23][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols22 fromPrev;
					channel_cols23 toNext;
					fromPrev = read_channel_intel(chain_output_channels22);
					#pragma unroll
					for (int col = 0; col < 23; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[23] = accumulation;
					write_channel_intel(chain_output_channels23, toNext);
					// printf ("[FPGA][PE23][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE24][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE24][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE24][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE24][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[24]);

					// printf ("[FPGA][PE24][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE24][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[25], feature);

					// printf ("[FPGA][PE24][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols23 fromPrev;
					channel_cols24 toNext;
					fromPrev = read_channel_intel(chain_output_channels23);
					#pragma unroll
					for (int col = 0; col < 24; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[24] = accumulation;
					write_channel_intel(chain_output_channels24, toNext);
					// printf ("[FPGA][PE24][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE25][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE25][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE25][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE25][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[25]);

					// printf ("[FPGA][PE25][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE25][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[26], feature);

					// printf ("[FPGA][PE25][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols24 fromPrev;
					channel_cols25 toNext;
					fromPrev = read_channel_intel(chain_output_channels24);
					#pragma unroll
					for (int col = 0; col < 25; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[25] = accumulation;
					write_channel_intel(chain_output_channels25, toNext);
					// printf ("[FPGA][PE25][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE26][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE26][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE26][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE26][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[26]);

					// printf ("[FPGA][PE26][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE26][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[27], feature);

					// printf ("[FPGA][PE26][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols25 fromPrev;
					channel_cols26 toNext;
					fromPrev = read_channel_intel(chain_output_channels25);
					#pragma unroll
					for (int col = 0; col < 26; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[26] = accumulation;
					write_channel_intel(chain_output_channels26, toNext);
					// printf ("[FPGA][PE26][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE27][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE27][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE27][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE27][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[27]);

					// printf ("[FPGA][PE27][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE27][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[28], feature);

					// printf ("[FPGA][PE27][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols26 fromPrev;
					channel_cols27 toNext;
					fromPrev = read_channel_intel(chain_output_channels26);
					#pragma unroll
					for (int col = 0; col < 27; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[27] = accumulation;
					write_channel_intel(chain_output_channels27, toNext);
					// printf ("[FPGA][PE27][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE28][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE28][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE28][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE28][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[28]);

					// printf ("[FPGA][PE28][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE28][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[29], feature);

					// printf ("[FPGA][PE28][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols27 fromPrev;
					channel_cols28 toNext;
					fromPrev = read_channel_intel(chain_output_channels27);
					#pragma unroll
					for (int col = 0; col < 28; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[28] = accumulation;
					write_channel_intel(chain_output_channels28, toNext);
					// printf ("[FPGA][PE28][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE29][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE29][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE29][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE29][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[29]);

					// printf ("[FPGA][PE29][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE29][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[30], feature);

					// printf ("[FPGA][PE29][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols28 fromPrev;
					channel_cols29 toNext;
					fromPrev = read_channel_intel(chain_output_channels28);
					#pragma unroll
					for (int col = 0; col < 29; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[29] = accumulation;
					write_channel_intel(chain_output_channels29, toNext);
					// printf ("[FPGA][PE29][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

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
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE30][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE30][%d] handling out channel %d\n", iter, out_ch);

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

		
			uint brick = 0;
			uint i = 0;
	
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE30][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE30][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[30]);

					// printf ("[FPGA][PE30][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE30][%d] Passing the feature to the next PE!\n", iter);
					write_channel_intel(chain_data_channels[31], feature);

					// printf ("[FPGA][PE30][%d] Done passing the feature to the next PE!\n", iter);
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt-1) {
					channel_cols29 fromPrev;
					channel_cols30 toNext;
					fromPrev = read_channel_intel(chain_output_channels29);
					#pragma unroll
					for (int col = 0; col < 30; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[30] = accumulation;
					write_channel_intel(chain_output_channels30, toNext);
					// printf ("[FPGA][PE30][%d] written something to the output channel!\n", iter);
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
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


	int iter = 0;

	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[31]);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		char frac_w = inst.frac_w;
		char frac_din = inst.frac_din;
		char frac_dout = inst.frac_dout;
		char out_ch_per_pe = inst.out_ch_per_pe;
		uint num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE31][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);

		char out_ch = 0;

		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE31][%d] handling out channel %d\n", iter, out_ch);

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

			uint brick = 0;
			uint i = 0;
			
			w_data accumulation;
			//for (uint brick = 0; brick < num_bricks; brick++) {
			while (brick != num_bricks) {
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;

				// printf ("[FPGA][PE31][%d] Handling brick %d\n", iter, brick);

				// Now it's time to read the inputs and do the calculation
				//for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE31][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(chain_data_channels[31]);

					// printf ("[FPGA][PE31][%d] Done waiting to read something!\n", iter);
					// Bypassing the data to next PE
					// printf ("[FPGA][PE31][%d] Passing the feature to the next PE!\n", iter);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				//}
				
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
				if (i == conv_loop_cnt - 1) {
					channel_cols30 fromPrev;
					channel_cols31 toNext;
					fromPrev = read_channel_intel(chain_output_channels30);
					#pragma unroll
					for (int col = 0; col < 31; col++) {
						toNext.cols[col] = fromPrev.cols[col];
					}
					toNext.cols[31] = accumulation;
					write_channel_intel(chain_output_channels31, toNext);
					// printf ("[FPGA][PE31][%d] written something to the output channel!\n", iter);
					
					#pragma unroll
					for (int wsize = 0; wsize < W_VEC; wsize++) {
						accumulation.w_data[wsize] = 0;
					}
					i = 0;
					brick++;
				} else {
					i++;
				}
			}
			out_ch++;
		}
		iter++;
	}
}


