__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memReadWeight(
			// Number of layers involved
			char config_size,
			// Params ports
			__global lane_cols	*restrict weights,
			__global channel_scal	*restrict biases)

{

	uint layer_offset = 0;

	for (char i = 0; i < config_size; i++) {

		memrd_weight_configuration config = read_channel_intel(memrd_weight_configuration_channel);

		int weight_m = config.weight_m;
		int weight_n = config.weight_n;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		ushort num_plates = weight_h * (weight_n/VEC_SIZE);

		uint offset = 0;

		for (ushort i = 0; i < weight_m; i+=LANE_NUM) {
			channel_scal bias_buffer;

			// Reading LANE_NUM of biases and send them to their 
			// respective PE
			bias_buffer = biases[i];

			for (char w = 0; w < LANE_NUM; w++) {
				// First send out the biases 
				write_channel_intel(chain_bias_channels[w], bias_buffer.lane[w]);

				// Now we read the weights and send them plate by plate to the 
				// appropriate PE
				for (ushort plate = 0; plate = num_plates; plate++) {
					lane_cols cur_plate = weights[layer_offset + offset];
					offset += 1;
					write_channel_intel(chain_weight_channels[w], cur_plate);
				}

			}
		}

		layer_offset += (weight_h * weight_n * weight_m) / VEC_SIZE;
	}

}