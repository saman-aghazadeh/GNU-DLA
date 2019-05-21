// Controller to dispatch computation, layer-by-layer
__kernel
__attribute__((task))
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
		int layer_type = configuration.layer_type;
		int data_w = configuration.data_w;
		int data_h = configuration.data_h;
		int weight_w = configuration.weight_w;
		int weight_h = configuration.weight_h;
		int weight_n = configuration.weight_n;
		int weight_m = configuration.weight_m;
		int bias_size = cconfiguration.bias_size;
		int memrd_src = configuration.memrd_src;
		int conv_x = configuration.conv_x;
		int conv_y = configuration.conv_y;
		int conv_z = configuration.conv_z;
		int conv_stride = configuration.conv_stride;
		int conv_padding = configuration.conv_padding;
		int conv_split = configuration.conv_split;
		int conv_relu = configuration.conv_relu;
		int pool_on = configuration.pool_on;
		int pool_x = configuration.pool_x;
		int pool_y = configuration.pool_y;
		int pool_z = configuration.pool_z;
		int pool_size = configuration.pool_size;
		int pool_stride = configuration.pool_stride;
		int lrn_on = configuration.lrn_on;
		int memwr_dst = configuration.memwr_dst;

		// This part controls the memrd module
		memrd_configuration memrd_config;
		memrd_config.layer_type = layer_type;
		memrd_config.data_w = data_w;
		memrd_config.data_h = data_h;

		write_channel_intel(memrd_configuration_channel, memrd_config);

		// This part controls the PEs
		instruction inst;
		inst.conv_loop_cnt = weight_h * (weight_n / VEC_SIZE);
		inst.frac_w = frac_w;
		inst.frac_dout = frac_dout;
		inst.frac_din = frac_din;
		write_channel_intel(chain_instruction_channels[0], inst);

		char done = 0;
		write_channel_intel(chain_done_layer_signal_channel[0], done);

		

	}


}