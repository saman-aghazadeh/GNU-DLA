// A final consumer for some of the signals

__kernel
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(1)))

void consumer() {

	while (1) {
		// int update_weights;
		// int done;
		lane_cols feature;
		instruction inst;
		// bias_DPTYPE bias;
		// weight_lane_cols weight;
		// update_weights = read_channel_intel(update_weights_signal_channel[LANE_NUM]);
		// done = read_channel_intel(chain_done_layer_signal_channel[LANE_NUM]);
		printf ("[FPGA][Consumer] Waiting for reading from the data_channel\n");
		feature = read_channel_intel(chain_data_channels[LANE_NUM]);
		printf ("[FPGA][Consumer] Done waiting for reading from the data_channel\n");
		inst = read_channel_intel(chain_instruction_channels[LANE_NUM]);
		printf ("[FPGA][Consumer] Done waiting for reading from the instruction channel\n");
		// bias = read_channel_intel(chain_bias_channels[LANE_NUM]);
		// weight = read_channel_intel(chain_weight_channels[LANE_NUM]);
	}
	
}
