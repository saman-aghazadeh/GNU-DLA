// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE0( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 0;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[0]);
			write_channel_intel(chain_done_layer_signal_channel[1], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[0]);
			write_channel_intel(update_weights_signal_channel[1], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE0[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[1], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[0]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[1], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE1( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 1;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[1]);
			write_channel_intel(chain_done_layer_signal_channel[2], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[1]);
			write_channel_intel(update_weights_signal_channel[2], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE1[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[2], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[1]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[2], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE2( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 2;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[2]);
			write_channel_intel(chain_done_layer_signal_channel[3], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[2]);
			write_channel_intel(update_weights_signal_channel[3], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE2[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[3], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[2]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[3], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE3( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 3;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[3]);
			write_channel_intel(chain_done_layer_signal_channel[4], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[3]);
			write_channel_intel(update_weights_signal_channel[4], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE3[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[4], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[3]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[4], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE4( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 4;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[4]);
			write_channel_intel(chain_done_layer_signal_channel[5], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[4]);
			write_channel_intel(update_weights_signal_channel[5], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE4[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[5], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[4]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[5], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE5( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 5;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[5]);
			write_channel_intel(chain_done_layer_signal_channel[6], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[5]);
			write_channel_intel(update_weights_signal_channel[6], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE5[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[6], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[5]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[6], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE6( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 6;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[6]);
			write_channel_intel(chain_done_layer_signal_channel[7], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[6]);
			write_channel_intel(update_weights_signal_channel[7], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE6[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[7], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[6]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[7], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE7( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 7;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[7]);
			write_channel_intel(chain_done_layer_signal_channel[8], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[7]);
			write_channel_intel(update_weights_signal_channel[8], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE7[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[8], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[7]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[8], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE8( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 8;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[8]);
			write_channel_intel(chain_done_layer_signal_channel[9], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[8]);
			write_channel_intel(update_weights_signal_channel[9], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE8[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[9], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[8]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[9], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE9( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 9;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[9]);
			write_channel_intel(chain_done_layer_signal_channel[10], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[9]);
			write_channel_intel(update_weights_signal_channel[10], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE9[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[10], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[9]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[10], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE10( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 10;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[10]);
			write_channel_intel(chain_done_layer_signal_channel[11], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[10]);
			write_channel_intel(update_weights_signal_channel[11], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE10[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[11], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[10]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[11], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE11( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 11;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[11]);
			write_channel_intel(chain_done_layer_signal_channel[12], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[11]);
			write_channel_intel(update_weights_signal_channel[12], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE11[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[12], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[11]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[12], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE12( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 12;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[12]);
			write_channel_intel(chain_done_layer_signal_channel[13], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[12]);
			write_channel_intel(update_weights_signal_channel[13], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE12[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[13], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[12]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[13], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE13( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 13;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[13]);
			write_channel_intel(chain_done_layer_signal_channel[14], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[13]);
			write_channel_intel(update_weights_signal_channel[14], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE13[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[14], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[13]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[14], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE14( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 14;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[14]);
			write_channel_intel(chain_done_layer_signal_channel[15], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[14]);
			write_channel_intel(update_weights_signal_channel[15], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE14[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[15], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[14]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[15], fromPrevToNext);
			}
		}

	}

}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__kernel
__attribute__((max_global_work_dim(0)))
__kernel void PE15( 
		// Number of layers involved
		char config_size,
		// Param ports
		__global lane_cols		*restrict weights,
		__global DPTYPE			*restrict biases 

) {

	char id = 15;
	//char bias_index = 0;
	//int weight_index = 0;

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	//__local lane_cols weight_buffer[WEIGHT_BUF_SIZE];

	//DPTYPE bias;

	//#pragma unroll 1
        //for (char i = 0; i < WEIGHT_BUF_SIZE; i++) {
        //        lane_cols temp_weight = weights[weight_index+i];
        //        weight_buffer[i] = temp_weight;
     	//}

        //weight_index += WEIGHT_BUF_SIZE;

	// Reading the bias and then 
	//bias = biases[bias_index];
	//bias_index += 1;

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

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		char num_weight_plates = inst.num_weight_plates;

		// All the work that should be done in this layer
		while (true) {

			int done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[15]);
			write_channel_intel(chain_done_layer_signal_channel[16], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			int update_weights_signal = read_channel_intel(update_weights_signal_channel[15]);
			write_channel_intel(update_weights_signal_channel[16], update_weights_signal);

			// We have to load the weights into the weight_buffer. weights
			// are loaded directly from the global memory. 
			// Hail mother mary, thou shall not fail us.
			// Thou shall not consume much MLABs.
			// Thou shall not consume much ALUs.
			// Thou shall not consume much RAMs.
			// Thou shall not perform poor.
			// Thou shall not act like a jerk.
			// Prais lord. Hallelujah!!!!!
			/*
			if (update_weights_signal == 0x01) {

				// Reading the bias and then 
				bias = biases[bias_index];
				bias_index += 1;

			}
			*/
			
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
						(accumulation.w_data[w]) + mac(feature.cols[w], weight_PE15[i].cols[w]);
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
			if (id == 0) {
				channel_cols toNext;
				toNext.cols[0] = accumulation;
				write_channel_intel(chain_output_channels[16], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_output_channels[15]);
				fromPrevToNext.cols[id] = accumulation;
				write_channel_intel(chain_output_channels[16], fromPrevToNext);
			}
		}

	}

}


