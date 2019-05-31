__kernel
__attribute__((max_global_work_dim(0)))
void memReadWeight(
			// Number of layers involved
			char config_size,
			// Params ports
			__global lane_cols	*restrict weights,
			__global bias_DPTYPE	*restrict biases)

{

	uint layer_offset = 0;

	for (char i = 0; i < config_size; i++) {

		memrd_weight_configuration config = read_channel_intel(memrd_weight_configuration_channel);

		int weight_m = config.weight_m;
		int weight_n = config.weight_n;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		ushort num_plates = weight_h * (weight_n/VEC_SIZE) * LANE_NUM;

		uint offset = 0;

		// We assume weight_m is divisible by LANE_NUM
		for (ushort i = 0; i < weight_m/LANE_NUM; i++) {
			bias_DPTYPE bias_buffer;
			// Reading LANE_NUM of biases and send them to their 
			// respective PE
			bias_buffer = biases[i];

			// First send out the biases
			// Case 1: 
			// write_channel_intel(chain_bias_channels[0], bias_buffer);
			// Case 2:
			for (int pe = 0; pe < LANE_NUM; pe++) {
				write_channel_intel(bias_channels[pe], bias_buffer.bias[pe]);
			}

			// Now we read the weights and send them plate by plate to the 
			// appropriate PE
			for (ushort plate = 0; plate < num_plates; plate++) {
				for (ushort pe = 0; pe < LANE_NUM; pe++) {
					lane_cols cur_plate = weights[offset];
					offset += 1;
					write_channel_intel(weight_channels[pe], cur_plate);
				}

				// Case 1: send the whole thing to the first PE,
				// and it will be bypassed to the second one and
				// the next PEs as well.
				// write_channel_intel(chain_weight_channels[0], cur_plate);
			
				// Case 2: There are dedicated channels to each PE,
				// so we will send the data to each of them, one by one.
				// for (int pe = 0; pe < LANE_NUM; pe++) {
				//	write_channel_intel(weight_channels[pe], cur_plate.weight[pe]);
				// }	
			}

		}

		//layer_offset += (weight_h * weight_n * weight_m) / VEC_SIZE;
	}

}
