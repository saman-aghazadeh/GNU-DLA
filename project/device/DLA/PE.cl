// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(LANE_NUM)))
__kernel void PE() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	lane_cols weight_buffer[WEIGHT_BUF_SIZE];
	DTYPE bias;
	int id = get_compute_id(0);


	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(chain_instruction_channels[id]);

		// Bypassing the instruction to the next PE
		write_channel_intel(chain_instruction_channels[id+1], inst);

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
		char num_weight_vecs = inst.num_weight_vecs;

		// All the work that should be done in this layer
		while (true) {

			char done_layer_signal = read_channel_intel(chain_done_layer_signal_channel[id]);
			write_channel_intel(chain_done_layer_signal_channel[id+1], done_layer_signal);
			if (done_layer_signal == 0x01) break;


			char update_weights_signal = read_channel_intel(chain_update_weights_signal_channel[id]);
			write_channel_intel(chain_update_weights_signal_channel[id+1], update_weights_signal)''

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			if (update_weights_signal == 0x01) {
				bias = read_channel_intel(chain_bias_channels[id]);
				for (char i = 0; i < num_weight_vecs; i++) {
					lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
					weight_buffer[i] = temp_weight;
				}
			}

			MACTYPE accumulation[W_VEC];
			MACTYPE acc_sign_exten[W_VEC];
			MACTYPE acc_with_rnd_bit[W_VEC];
			MACTYPE acc_sum_bias[W_VEC];

			// Now it's time to read the inputs and do the calculation
			for (uint i = 0; i < conv_loop_cnt; i++) {
				// Reading data incoming feature data from the incoming input
				lane_cols feature = read_channel_intel(chain_data_channels[id]);

				// Bypassing the data to next PE
				write_channel_intel(chain_data_channels[id+1], feature);

				#pragma unroll
				for (char w = 0; w < W_VEC; w++) {
					accumulation[w] = 
						(MASK_MULT & accumulation[w]) + 
						MASK_MULT & mac(feature.cols[w], weight_buffer[i].cols[w]);
				}
			}


			// Not sure why we have to do all these
			#pragma unroll
			for (unsigned i = 0; i < W_VEC; i++) {
				if (accumulation[i] > 0)
					acc_sign_exten[i] = 0x00;
				else
					acc_sign_exten[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

				acc_with_rnd_bit[i] = (acc_sign_exten[i] | (accumulation[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

				// This part should be fixed
				if (acc_with_rnd_bit[i] >= 256)
					acc_sum_bias[i] = MASK9B & 0xFF;
				else if (acc_with_rnd_bit[i] < -256)
					acc_sum_bias[i] = MASK9B & 0x100;
				else
					acc_sum_bias[i] = (MASK9B & acc_with_rnd_bit[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

				accumulation[i] = MASK8B & (acc_sum_bias[i] >> 0x01);

				
			}

			// After an array of output is generated, we will bypass that output to the next layer.
			// It actually combines it's own output with the previous layer output, and then pass 
			// it on
			if (id == 0) {
				channel_cols toNext;
				channel_cols.cols[0] = accumulation;
				write_channel_intel(chain_done_layer_signal_channel[id+1], toNext);
			} else {
				channel_cols fromPrevToNext;
				fromPrevToNext = read_channel_intel(chain_done_layer_signal_channel[id]);
				fromPrevToNext[id] = accumulation;
				write_channel_intel(chain_done_layer_signal_channel[id+1], fromPrevToNext);
			}
		}

	}

}