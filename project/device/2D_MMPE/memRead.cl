channel lane_cols					mem_read_data0_channels;
channel lane_cols					mem_read_data1_channels;
channel inv_rows					mem_write_data0_channels;
channel inv_rows					mem_write_data1_channels;

#define WIN_BUF_SIZE0		(16384*8)
#define WIN_BUF_SIZE1		(32768*8)

typedef struct
{
	int total_num_plates;
}router_configuration;

channel router_configuration 		memrd_router_configuration_channel;
channel router_configuration 		memwrite_router_configuration_channel;

// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))

void memReadData(
		// Number of layers involved
		char config_size,
		// Data Ports
		__global lane_data	*restrict bottom0,
		__global lane_data 	*restrict bottom1) {

	// This flag specifies which global data part we should
	// read the data from.
	char flag = 0x00;
	lane_data data_buffer[WIN_BUF_SIZE0 * W_VEC / VEC_SIZE];
	for(int l = 0; l < WIN_BUF_SIZE0 * W_VEC / VEC_SIZE; l++) {
		data_buffer[l] = bottom0[l];
	}

	for (char i = 0; i < config_size; i++) {

		// Reading the configuration for the specific layer
		memrd_data_configuration config = read_channel_intel(memrd_data0_configuration_channel);
		int layer_type = config.layer_type;
		int data_w = config.data_w;
		int data_h = config.data_h;
		int data_t = config.data_t;
		int weight_m = config.weight_m;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		int weight_n = config.weight_n;
		int weight_t = config.weight_t;
		int conv_stride = config.conv_stride;
		int conv_padding = config.conv_padding;
		int conv_x = config.conv_x;
		int conv_y = config.conv_y;
		int conv_z = config.conv_z;
		int conv_t = config.conv_t;

		int data_w_with_padding = data_w + 2 * conv_padding;
		int data_h_with_padding = data_h + 2 * conv_padding;
		int data_t_with_padding = data_t + 2 * conv_padding;

		//if (i >= 13)
			// printf ("[FPGA][memReadData][%d] layer_type=%d, data_w=%d, data_h=%d, data_t=%d, weight_m=%d, weight_h=%d, weight_w=%d, weight_n=%d, weight_t=%d, conv_padding=%d, data_w_with_padding=%d, data_h_with_padding=%d\n", i, layer_type, data_w, data_h, data_t, weight_m, weight_h, weight_w, weight_n, weight_t, conv_padding, data_w_with_padding, data_h_with_padding);

		// It may seems strange, but it's memReadData responsibility, to let the 
		// PE knows that it has to load a new set of weights.

		// TODO: We assume for now that weight_m is divisble by LANE_NUM
		int out_channel_iter = weight_m / LANE_NUM;
		//if (i >= 13)
		//	printf ("[FPGA][memReadData][%d] out_channel_iter is %d\n", i, out_channel_iter);	

		if((i & 0x01) == 0x00) {
			uint num_bricks;
			for (int j = 0; j < out_channel_iter; j++) {
	
			//if (i >= 13)
			//	printf ("[FPGA][memReadData][%d] processing out channel=%d\n", i, j*LANE_NUM);
			// We have to read the data brick by brick.
			// Every brick is of size 
			// W_VEC * weight_h * weight_n
			// weight_w has been replaced with W_VEC, since we are
			// going to do winograd convolution :D

			ushort brick_idx_x = 0;
			ushort brick_idx_y = 0;
			ushort brick_idx_t = 0;
			num_bricks = 0;
			while (brick_idx_t != data_t_with_padding-weight_t+1) {

				//if (i >= 13)
					// printf ("[FPGA][memReadData][%d] Processing a new brick with brick_idx_x=%d, brick_idx_y=%d and brick_idx_t=%d and id=%d\n", i, brick_idx_x, brick_idx_y, brick_idx_t, num_bricks);

				// These indexes determines where are we in the 
				// feature map.
				char feature_idx_x = brick_idx_x;
				char feature_idx_y = brick_idx_y;
				char feature_idx_t = brick_idx_t;
				char feature_idx_z = 0;

				// TODO: Here assume weight_n is divisible by VEC_SIZE
				short num_plates = weight_h * (weight_n/VEC_SIZE) * weight_t;

				//if (i >= 13)
				//	printf ("[FPGA][memReadData][%d] num_plates is %d\n", i, num_plates);

				for (short plate = 0; plate < num_plates; plate++) {
					//if (i >= 12)
						//printf ("[FPGA][memReadData] Processing plate=%d\n", plate);						
					lane_cols data_for_convs;

					// We have to read a plate of data, which is of 
					// size W_VEC * VEC_SIZE
					short read_index = 
						feature_idx_t * data_w * data_h * (weight_n/VEC_SIZE) +
						feature_idx_z * data_w * data_h +
						(feature_idx_y-conv_padding) * data_w +
						(feature_idx_x-conv_padding);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {

						// if ((feature_idx_x+w >= conv_padding && feature_idx_x+w < data_w + conv_padding)
						//	&&
						//	(feature_idx_y >= conv_padding && feature_idx_y < data_h + conv_padding)) {
							// if (flag == 0x00) {
								// data_for_convs.cols[w] = data_buffer[read_index+w];
							// } else {
								// data_for_convs.cols[w] = bottom1[read_index+w];
							// }
						// } else {
						//	#pragma unroll
						//	for (unsigned char vv = 0; vv < VEC_SIZE; vv++)
						//		data_for_convs.cols[w].data[vv] = CZERO;
						// }
					}

					// DAMN we read the plate. Now it's time for peanut butter jelly.
					// Just kidding! we have to send the data to the first PE. 
					//if (i >= 12)
					//	printf ("[FPGA][memReadData][%d] Before sending the data\n", i);
					write_channel_intel(mem_read_data0_channels, data_for_convs);
					//if (i >= 12)
					//	printf ("[FPGA][memReadData][%d] After sending the data\n", i);

					// Alright data is sent, we have to move on to the next plate.
					// That means, we have to update out feature indexes

					if ((feature_idx_t == weight_t-1) && (feature_idx_z == weight_n/VEC_SIZE-1) && (feature_idx_y == weight_h-1)) {
						feature_idx_t = 0;
					} else if ((feature_idx_z == weight_n/VEC_SIZE-1) && (feature_idx_y == weight_h-1)) {
						feature_idx_t++;
					}					

					if ((feature_idx_z == weight_n/VEC_SIZE-1) && (feature_idx_y == weight_h-1)) {
						feature_idx_z = 0;
					} else if (feature_idx_y == weight_h-1){
						feature_idx_z++;
					}

					if (feature_idx_y == weight_h-1) {
						feature_idx_y = 0;
					} else {
						feature_idx_y++;
					}
				}

				//if ((brick_idx_y == data_h_with_padding-1) && (brick_idx_x+(W_VEC+1) > data_w_with_padding )){
				//	brick_idx_y++;
				//} else if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
				//	brick_idx_y++;
				//}

				if (((brick_idx_y == data_h_with_padding-weight_h)) && (brick_idx_x + (W_VEC+1) > data_w_with_padding)) {
					brick_idx_y = 0;
					brick_idx_t++;
				} else if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
					brick_idx_y++;
				}

				if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
					brick_idx_x = 0;
				} else {
					brick_idx_x += (W_VEC-weight_w+1);
				}

				num_bricks++;
				}
			}
			printf("[FPGA][memReadData0][%d] num_plates %d\n", i, out_channel_iter * num_bricks * weight_h * (weight_n/VEC_SIZE) * weight_t);
		}
		else {
			int write_index = 0;

			// We assume conv_z is divisble by LANE_NUM
			uint num_plates = conv_y * (conv_z/LANE_NUM) * conv_t * ((conv_x-1)/W_INV_VEC + 1);
			
			//if (i >= 13)
				printf ("[FPGA][memWrite0][%d] conv_x=%d, conv_y=%d, conv_z=%d, weight_w=%d, num_plates=%d\n", i, conv_x, conv_y, conv_z, weight_w, num_plates);

			for (uint plate = 0; plate < num_plates; plate++) {
				inv_rows inv;

				inv = read_channel_intel(mem_write_data0_channels);

				//if (i >= 13)
				//	printf ("[FPGA][memWrite][%d] plate=%d\n", i, plate);
				
				//if (i >= 13)
				//	printf ("[FPGA][memWrite][%d] start writing to memory!\n", i);
				#pragma unroll
				for (char l = 0; l < LANE_NUM; l++) {
					#pragma unroll
					for (char w = 0; w < W_INV_VEC; w++) {
						// data_buffer[write_index+w+(l/VEC_SIZE)].data[l%VEC_SIZE] = inv.cols[w].lane[l];
					}
				}

				//if (i >= 12)
				//	printf ("[FPGA][memWrite][%d] finished writing to memory!\n", i);
				write_index += W_INV_VEC * (LANE_NUM / VEC_SIZE);
			}
		}

		// Now we are swapping to the alternative buffer, which contains
		// the data for the next layer
		flag = (~flag) & 0x01;
	}
	for(int l = 0; l < WIN_BUF_SIZE0 * W_VEC / VEC_SIZE; l++) {
		bottom1[l] = data_buffer[l];
	}
}

// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))

void memWrite(
		// Number of layers involved
		char config_size,
		// Data Ports
		__global lane_data	*restrict bottom0,
		__global lane_data 	*restrict bottom1) {

	// This flag specifies which global data part we should
	// read the data from.
	char flag = 0x00;
	lane_data data_buffer[WIN_BUF_SIZE1 * W_VEC / VEC_SIZE];
	for(int l = 0; l < WIN_BUF_SIZE1 * W_VEC / VEC_SIZE; l++) {
		data_buffer[l] = bottom0[l];
	}
	for (char i = 0; i < config_size; i++) {

		// Reading the configuration for the specific layer
		memrd_data_configuration config = read_channel_intel(memrd_data1_configuration_channel);
		int layer_type = config.layer_type;
		int data_w = config.data_w;
		int data_h = config.data_h;
		int data_t = config.data_t;
		int weight_m = config.weight_m;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		int weight_n = config.weight_n;
		int weight_t = config.weight_t;
		int conv_stride = config.conv_stride;
		int conv_padding = config.conv_padding;
		int conv_x = config.conv_x;
		int conv_y = config.conv_y;
		int conv_z = config.conv_z;
		int conv_t = config.conv_t;

		int data_w_with_padding = data_w + 2 * conv_padding;
		int data_h_with_padding = data_h + 2 * conv_padding;
		int data_t_with_padding = data_t + 2 * conv_padding;

		//if (i >= 13)
			// printf ("[FPGA][memReadData][%d] layer_type=%d, data_w=%d, data_h=%d, data_t=%d, weight_m=%d, weight_h=%d, weight_w=%d, weight_n=%d, weight_t=%d, conv_padding=%d, data_w_with_padding=%d, data_h_with_padding=%d\n", i, layer_type, data_w, data_h, data_t, weight_m, weight_h, weight_w, weight_n, weight_t, conv_padding, data_w_with_padding, data_h_with_padding);

		// It may seems strange, but it's memReadData responsibility, to let the 
		// PE knows that it has to load a new set of weights.

		// TODO: We assume for now that weight_m is divisble by LANE_NUM
		int out_channel_iter = weight_m / LANE_NUM;
		//if (i >= 13)
		//	printf ("[FPGA][memReadData][%d] out_channel_iter is %d\n", i, out_channel_iter);	
		
		if((i & 0x01) == 0x01) {
			uint num_bricks;
			for (int j = 0; j < out_channel_iter; j++) {
		
				//if (i >= 13)
				//	printf ("[FPGA][memReadData][%d] processing out channel=%d\n", i, j*LANE_NUM);
				// We have to read the data brick by brick.
				// Every brick is of size 
				// W_VEC * weight_h * weight_n
				// weight_w has been replaced with W_VEC, since we are
				// going to do winograd convolution :D

				ushort brick_idx_x = 0;
				ushort brick_idx_y = 0;
				ushort brick_idx_t = 0;
				num_bricks = 0;
				while (brick_idx_t != data_t_with_padding-weight_t+1) {

					//if (i >= 13)
						// printf ("[FPGA][memReadData][%d] Processing a new brick with brick_idx_x=%d, brick_idx_y=%d and brick_idx_t=%d and id=%d\n", i, brick_idx_x, brick_idx_y, brick_idx_t, num_bricks);

					// These indexes determines where are we in the 
					// feature map.
					char feature_idx_x = brick_idx_x;
					char feature_idx_y = brick_idx_y;
					char feature_idx_t = brick_idx_t;
					char feature_idx_z = 0;

					// TODO: Here assume weight_n is divisible by VEC_SIZE
					short num_plates = weight_h * (weight_n/VEC_SIZE) * weight_t;

					//if (i >= 13)
					//	printf ("[FPGA][memReadData][%d] num_plates is %d\n", i, num_plates);

					for (short plate = 0; plate < num_plates; plate++) {
						//if (i >= 12)
							//printf ("[FPGA][memReadData] Processing plate=%d\n", plate);						
						lane_cols data_for_convs;

						// We have to read a plate of data, which is of 
						// size W_VEC * VEC_SIZE
						short read_index = 
							feature_idx_t * data_w * data_h * (weight_n/VEC_SIZE) +
							feature_idx_z * data_w * data_h +
							(feature_idx_y-conv_padding) * data_w +
							(feature_idx_x-conv_padding);

						#pragma unroll
						for (char w = 0; w < W_VEC; w++) {

							// if ((feature_idx_x+w >= conv_padding && feature_idx_x+w < data_w + conv_padding)
							//	&&
							//	(feature_idx_y >= conv_padding && feature_idx_y < data_h + conv_padding)) {
								// if (flag == 0x00) {
									// data_for_convs.cols[w] = data_buffer[read_index+w];
								// } else {
									// data_for_convs.cols[w] = bottom1[read_index+w];
								// }
							// } else {
							//	#pragma unroll
							//	for (unsigned char vv = 0; vv < VEC_SIZE; vv++)
							//		data_for_convs.cols[w].data[vv] = CZERO;
							// }
						}

						// DAMN we read the plate. Now it's time for peanut butter jelly.
						// Just kidding! we have to send the data to the first PE. 
						//if (i >= 12)
						//	printf ("[FPGA][memReadData][%d] Before sending the data\n", i);
						write_channel_intel(mem_read_data1_channels, data_for_convs);
						//if (i >= 12)
						//	printf ("[FPGA][memReadData][%d] After sending the data\n", i);

						// Alright data is sent, we have to move on to the next plate.
						// That means, we have to update out feature indexes

						if ((feature_idx_t == weight_t-1) && (feature_idx_z == weight_n/VEC_SIZE-1) && (feature_idx_y == weight_h-1)) {
							feature_idx_t = 0;
						} else if ((feature_idx_z == weight_n/VEC_SIZE-1) && (feature_idx_y == weight_h-1)) {
							feature_idx_t++;
						}					

						if ((feature_idx_z == weight_n/VEC_SIZE-1) && (feature_idx_y == weight_h-1)) {
							feature_idx_z = 0;
						} else if (feature_idx_y == weight_h-1){
							feature_idx_z++;
						}

						if (feature_idx_y == weight_h-1) {
							feature_idx_y = 0;
						} else {
							feature_idx_y++;
						}
					}

					//if ((brick_idx_y == data_h_with_padding-1) && (brick_idx_x+(W_VEC+1) > data_w_with_padding )){
					//	brick_idx_y++;
					//} else if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
					//	brick_idx_y++;
					//}

					if (((brick_idx_y == data_h_with_padding-weight_h)) && (brick_idx_x + (W_VEC+1) > data_w_with_padding)) {
						brick_idx_y = 0;
						brick_idx_t++;
					} else if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
						brick_idx_y++;
					}

					if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
						brick_idx_x = 0;
					} else {
						brick_idx_x += (W_VEC-weight_w+1);
					}

					num_bricks++;
				}
			}
			printf("[FPGA][memReadData1][%d] num_plates %d\n", i, out_channel_iter * num_bricks * weight_h * (weight_n/VEC_SIZE) * weight_t);
		}
		else {
			int write_index = 0;

			// We assume conv_z is divisble by LANE_NUM
			uint num_plates = conv_y * (conv_z/LANE_NUM) * conv_t * ((conv_x-1)/W_INV_VEC + 1);
			
			//if (i >= 13)
				printf ("[FPGA][memWrite1][%d] conv_x=%d, conv_y=%d, conv_z=%d, weight_w=%d, num_plates=%d\n", i, conv_x, conv_y, conv_z, weight_w, num_plates);

			for (uint plate = 0; plate < num_plates; plate++) {
				inv_rows inv;

				inv = read_channel_intel(mem_write_data1_channels);

				//if (i >= 13)
				//	printf ("[FPGA][memWrite][%d] plate=%d\n", i, plate);
				
				//if (i >= 13)
				//	printf ("[FPGA][memWrite][%d] start writing to memory!\n", i);
				#pragma unroll
				for (char l = 0; l < LANE_NUM; l++) {
					#pragma unroll
					for (char w = 0; w < W_INV_VEC; w++) {
						// data_buffer[write_index+w+(l/VEC_SIZE)].data[l%VEC_SIZE] = inv.cols[w].lane[l];
					}
				}

				//if (i >= 12)
				//	printf ("[FPGA][memWrite][%d] finished writing to memory!\n", i);
				write_index += W_INV_VEC * (LANE_NUM / VEC_SIZE);
			}
		}
		// Now we are swapping to the alternative buffer, which contains
		// the data for the next layer
		flag = (~flag) & 0x01;
	}
	for(int l = 0; l < WIN_BUF_SIZE1 * W_VEC / VEC_SIZE; l++) {
		bottom1[l] = data_buffer[l];
	}
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void memread_router() {
	char flag = 0;
	while(true) {
		router_configuration memrd_router_config = read_channel_intel(memrd_router_configuration_channel);
		int total_num_plates = memrd_router_config.total_num_plates;
		for(int plate = 0; plate < total_num_plates; plate++) {
			lane_cols data_for_convs;
			if((flag & 0x01) == 0x00)
				data_for_convs = read_channel_intel(mem_read_data0_channels);
			else
				data_for_convs = read_channel_intel(mem_read_data1_channels);
			write_channel_intel(winograd_transform_channels, data_for_convs);
		}
		flag = (~flag) & 0x01;
	}
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void memwrite_router() {
	char flag = 0;
	while(true) {
		router_configuration memwrite_router_config = read_channel_intel(memwrite_router_configuration_channel);
		int total_num_plates = memwrite_router_config.total_num_plates;
		for(int plate = 0; plate < total_num_plates; plate++) {
			inv_rows inv = read_channel_intel(winograd_inv_transform_channels);
			if((flag & 0x01) == 0x01)
				write_channel_intel(mem_write_data0_channels, inv);
			else
				write_channel_intel(mem_write_data1_channels, inv) ;
		}
		flag = (~flag) & 0x01;
	}
}

// Controller to dispatch computation, layer-by-layer
__kernel
__attribute__((max_global_work_dim(0)))
void controller(
	// Number of layers involved
	char config_size,
	// Parameters
	// char frac_w,
	// char frac_din,
	// char frac_dout,
	// Configuration parameters
	__global configuration	*restrict config) {

	// printf ("[FPGA][Controller] Number of layers is %d\n", config_size);
	char frac_w = 7;
	char frac_din = 0;
	char frac_dout = 2;
	
	for (int i = 0; i < config_size; i++) {

		int layer_type = config[i].layer_type;
		int data_w = config[i].data_w;
		int data_h = config[i].data_h;
		int data_t = config[i].data_t;
		int weight_w = config[i].weight_w;
		int weight_h = config[i].weight_h;
		int weight_n = config[i].weight_n;
		int weight_t = config[i].weight_t;
		int weight_m = config[i].weight_m;
		int bias_size = config[i].bias_size;
		int memrd_src = config[i].memrd_src;
		int conv_x = config[i].conv_x;
		int conv_y = config[i].conv_y;
		int conv_z = config[i].conv_z;
		int conv_t = config[i].conv_t;
		int conv_stride = config[i].conv_stride;
		int conv_padding = config[i].conv_padding;
		int conv_split = config[i].conv_split;
		int conv_relu = config[i].conv_relu;
		int pool_on = config[i].pool_on;
		int pool_x = config[i].pool_x;
		int pool_y = config[i].pool_y;
		int pool_z = config[i].pool_z;
		int pool_t = config[i].pool_t;
		int pool_size_xy = config[i].pool_size_xy;
		int pool_size_t = config[i].pool_size_t;
		int pool_stride_xy = config[i].pool_stride_xy;
		int pool_stride_t = config[i].pool_stride_t;
		int lrn_on = config[i].lrn_on;
		int memwr_dst = config[i].memwr_dst;
		int num_bricks = config[i].num_bricks;
		int num_weight_plates = weight_h * (weight_n/VEC_SIZE) * weight_t;
		int out_ch_per_pe = weight_m / LANE_NUM;
		//if (i >= 12) {
			// printf ("[FPGA][Controller] Layer %d execution. layer_type=%d, data_w=%d, data_h=%d, weight_w=%d, weight_h=%d, weight_n=%d, weight_m=%d, bias_size=%d, memrd_src=%d, conv_x=%d, conv_y=%d, conv_z=%d, num_bricks=%d\n", i, layer_type, data_w, data_h, weight_w, weight_h, weight_n, weight_m, bias_size, memrd_src, conv_x, conv_y, conv_z, num_bricks);
		//}

		if(layer_type == 0) {
			// This part controls the memrd_data module
			memrd_data_configuration memrd_data_config;
			memrd_data_config.layer_type = layer_type;
			memrd_data_config.data_w = data_w;
			memrd_data_config.data_h = data_h;
			memrd_data_config.data_t = data_t;
			memrd_data_config.weight_m = weight_m;
			memrd_data_config.weight_h = weight_h;
			memrd_data_config.weight_w = weight_w;
			memrd_data_config.weight_n = weight_n;
			memrd_data_config.weight_t = weight_t;
			memrd_data_config.conv_padding = conv_padding;
			memrd_data_config.conv_stride = conv_stride;
			memrd_data_config.conv_x = conv_x;
			memrd_data_config.conv_y = conv_y;
			memrd_data_config.conv_z = conv_z;
			memrd_data_config.conv_t = conv_t;
			write_channel_intel(memrd_data0_configuration_channel, memrd_data_config);
			write_channel_intel(memrd_data1_configuration_channel, memrd_data_config);

			router_configuration memrd_router_config;
			memrd_router_config.total_num_plates = out_ch_per_pe * num_bricks * num_weight_plates;
			write_channel_intel(memrd_router_configuration_channel, memrd_router_config);		

			router_configuration memwrite_router_config;
			memwrite_router_config.total_num_plates = conv_y * (conv_z/LANE_NUM) * conv_t * ((conv_x-1)/W_INV_VEC + 1);
			write_channel_intel(memwrite_router_configuration_channel, memwrite_router_config);		

			// This part controls the memrd_weight module
			memrd_weight_configuration memrd_weight_config;
			memrd_weight_config.weight_m = weight_m;
			memrd_weight_config.weight_n = weight_n;
			memrd_weight_config.weight_h = weight_h;
			memrd_weight_config.weight_w = weight_w;
			memrd_weight_config.weight_t = weight_t;
			write_channel_intel(memrd_weight_configuration_channel, memrd_weight_config);	

			// This part controls the PEs
			instruction inst;
			inst.conv_loop_cnt = num_weight_plates;
			inst.frac_w = frac_w;
			inst.frac_dout = frac_dout;
			inst.frac_din = frac_din;
			inst.num_weight_plates = num_weight_plates;
			inst.out_ch_per_pe = out_ch_per_pe;
			inst.num_bricks = num_bricks;
			write_channel_intel(chain_instruction_channels[0], inst);

			// This part controls the memwr module
			// memwr_configuration memwr_config;
			// memwr_config.conv_x = conv_x;
			// memwr_config.conv_y = conv_y;
			// memwr_config.conv_z = conv_z;
			// memwr_config.conv_t = conv_t;
			// memwr_config.weight_w = weight_w;
			// write_channel_intel(memwr_configuration_channel, memwr_config);
		}
		else {
			// This part controls the memrd_weight module
			memrd_weight_configuration memrd_weight_config;
			memrd_weight_config.weight_m = weight_m;
			memrd_weight_config.weight_n = weight_n;
			memrd_weight_config.weight_h = weight_h;
			memrd_weight_config.weight_w = weight_w;
			memrd_weight_config.weight_t = weight_t;
			write_channel_intel(fc_memrd_configuration_channel, memrd_weight_config);

			// This part controls the PEs
			instruction inst;
			inst.conv_loop_cnt = weight_h * (weight_n / VEC_SIZE) * weight_t;
			inst.frac_w = frac_w;
			inst.frac_dout = frac_dout;
			inst.frac_din = frac_din;
			inst.num_weight_plates = weight_h * (weight_n/VEC_SIZE) * weight_t;
			inst.out_ch_per_pe = weight_m;
			inst.num_bricks = num_bricks;
			write_channel_intel(fc_instruction_channel, inst);

			// This part controls the memwr module
			memwr_configuration memwr_config;
			memwr_config.conv_x = conv_x;
			memwr_config.conv_y = conv_y;
			memwr_config.conv_z = conv_z;
			memwr_config.conv_t = conv_t;
			memwr_config.weight_w = weight_w;
			write_channel_intel(fc_memwr_configuration_channel, memwr_config);
		}
	}
}
