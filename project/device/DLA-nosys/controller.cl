// Controller to dispatch computation, layer-by-layer
__kernel
__attribute__((max_global_work_dim(0)))

void controller(
	// Number of layers involved
	char config_size,
	// Parameters
	char frac_w,
	char frac_din,
	char frac_dout,
	// Configuration parameters
	__global configuration	*restrict config) 

{

	// printf ("[FPGA][Controller] Number of layers is %d\n", config_size);

	for (int i = 0; i < config_size; i++) {

		int layer_type = config[i].layer_type;
		int data_w = config[i].data_w;
		int data_h = config[i].data_h;
		int weight_w = config[i].weight_w;
		int weight_h = config[i].weight_h;
		int weight_n = config[i].weight_n;
		int weight_m = config[i].weight_m;
		int bias_size = config[i].bias_size;
		int memrd_src = config[i].memrd_src;
		int conv_x = config[i].conv_x;
		int conv_y = config[i].conv_y;
		int conv_z = config[i].conv_z;
		int conv_stride = config[i].conv_stride;
		int conv_padding = config[i].conv_padding;
		int conv_split = config[i].conv_split;
		int conv_relu = config[i].conv_relu;
		int pool_on = config[i].pool_on;
		int pool_x = config[i].pool_x;
		int pool_y = config[i].pool_y;
		int pool_z = config[i].pool_z;
		int pool_size = config[i].pool_size;
		int pool_stride = config[i].pool_stride;
		int lrn_on = config[i].lrn_on;
		int memwr_dst = config[i].memwr_dst;
		int num_bricks = config[i].num_bricks;

		//if (i >= 12) {
		//	printf ("[FPGA][Controller] Layer %d execution. layer_type=%d, data_w=%d, data_h=%d, weight_w=%d, weight_h=%d, weight_n=%d, weight_m=%d, bias_size=%d, memrd_src=%d, conv_x=%d, conv_y=%d, conv_z=%d, num_bricks=%d\n", i, layer_type, data_w, data_h, weight_w, weight_h, weight_n, weight_m, bias_size, memrd_src, conv_x, conv_y, conv_z, num_bricks);
		//}

		// This part controls the memrd_data module
		memrd_data_configuration memrd_data_config;
		memrd_data_config.layer_type = layer_type;
		memrd_data_config.data_w = data_w;
		memrd_data_config.data_h = data_h;
		memrd_data_config.weight_m = weight_m;
		memrd_data_config.weight_h = weight_h;
		memrd_data_config.weight_w = weight_w;
		memrd_data_config.weight_n = weight_n;
		memrd_data_config.conv_padding = conv_padding;
		write_channel_intel(memrd_data_configuration_channel, memrd_data_config);

		// This part controls the memrd_weight module
		memrd_weight_configuration memrd_weight_config;
		memrd_weight_config.weight_m = weight_m;
		memrd_weight_config.weight_n = weight_n;
		memrd_weight_config.weight_h = weight_h;
		memrd_weight_config.weight_w = weight_w;
		write_channel_intel(memrd_weight_configuration_channel, memrd_weight_config);	

		// This part controls the PEs
		instruction inst;
		inst.conv_loop_cnt = weight_h * (weight_n / VEC_SIZE);
		inst.frac_w = frac_w;
		inst.frac_dout = frac_dout;
		inst.frac_din = frac_din;
		inst.num_weight_plates = weight_h * (weight_n/VEC_SIZE);
		inst.out_ch_per_pe = weight_m / LANE_NUM;
		inst.num_bricks = num_bricks;
		write_channel_intel(chain_instruction_channels[0], inst);

		// This part controls the memwr module
		memwr_configuration memwr_config;
		memwr_config.conv_x = conv_x;
		memwr_config.conv_y = conv_y;
		memwr_config.conv_z = conv_z;
		memwr_config.weight_w = weight_w;
		write_channel_intel(memwr_configuration_channel, memwr_config);

	}


}
