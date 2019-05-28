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

	for (int i = 0; i < config_size; i++) {
		int layer_type = config->layer_type;
		int data_w = config->data_w;
		int data_h = config->data_h;
		int weight_w = config->weight_w;
		int weight_h = config->weight_h;
		int weight_n = config->weight_n;
		int weight_m = config->weight_m;
		int bias_size = config->bias_size;
		int memrd_src = config->memrd_src;
		int conv_x = config->conv_x;
		int conv_y = config->conv_y;
		int conv_z = config->conv_z;
		int conv_stride = config->conv_stride;
		int conv_padding = config->conv_padding;
		int conv_split = config->conv_split;
		int conv_relu = config->conv_relu;
		int pool_on = config->pool_on;
		int pool_x = config->pool_x;
		int pool_y = config->pool_y;
		int pool_z = config->pool_z;
		int pool_size = config->pool_size;
		int pool_stride = config->pool_stride;
		int lrn_on = config->lrn_on;
		int memwr_dst = config->memwr_dst;

		// This part controls the memrd module
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

		// This part controls the PEs
		instruction inst;
		inst.conv_loop_cnt = weight_h * (weight_n / VEC_SIZE);
		inst.frac_w = frac_w;
		inst.frac_dout = frac_dout;
		inst.frac_din = frac_din;
		inst.num_weight_plates = weight_h * (weight_n/VEC_SIZE);
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
