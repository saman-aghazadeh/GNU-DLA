__kernel
__attribute__((max_global_work_dim(0)))
void memReadWeight(
			// Number of layers involved
			char config_size,
			// Params ports
			__global lane_cols	*restrict weights,
			__global bias_DPTYPE	*restrict biases)

{
	// printf ("[FPGA][memReadWeight] Number of layers is %d\n", config_size);

	uint layer_offset = 0;

	for (char i = 0; i < config_size; i++) {

		memrd_weight_configuration config = read_channel_intel(memrd_weight_configuration_channel);

		int weight_m = config.weight_m;
		int weight_n = config.weight_n;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		int weight_t = config.weight_t;
		int split_weight_n = config.split_weight_n;
		int split_weight_n_offset = config.split_weight_n_offset;
		ushort num_plates = weight_t * weight_h * (split_weight_n/VEC_SIZE);

		//if (i >= 13)
		// 	printf ("[FPGA][memReadWeight][%d] weight_m=%d, weight_n=%d, weight_h=%d, weight_w=%d, num_plates=%d\n", i, weight_m, weight_n, weight_h, weight_w, num_plates);

		// assuming that the weight for every sub-layer is ordered sequentially one after the other
		uint offset = split_weight_n_offset * weight_h * weight_t;

		// We assume weight_m is divisible by LANE_NUM
		for (ushort j = 0; j < weight_m/LANE_NUM; j++) {
			
			//if (i >= 13)
			// 	printf ("[FPGA][memReadWeight][%d] Processing output channel %d\n", i, j*LANE_NUM);

			bias_DPTYPE bias_buffer;
			// Reading LANE_NUM of biases and send them to their 
			// respective PE
			bias_buffer = biases[j];

			// Now we read the weights and send them plate by plate to the 
			// appropriate PEs
			for (ushort pe = 0; pe < LANE_NUM; pe++) {
				//if (i >= 13)
				//	printf ("[FPGA][memReadWeight][%d] Processing lane %d\n", i, pe);
				write_channel_intel(bias_channels[pe], bias_buffer.bias[pe]);
				for (ushort plate = 0; plate < num_plates; plate++) {
					//if (i >= 13)
					//	printf ("[FPGA][memReadWeight][%d] Processing plate %d\n", i, plate);
					lane_cols cur_plate = weights[offset];
					offset += 1;
					write_channel_intel(weight_channels[pe], cur_plate);
				}
			}
		}
	}
}
