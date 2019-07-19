// Store Data to Global Memory
__kernel
__attribute__((max_global_work_dim(0)))
void memWrite(
				char config_size,
				// Params Ports
				// Data Ports
                __global channel_scal *restrict bottom0,
                __global channel_scal *restrict bottom1
				)
{

	// This flag specified which global data part we should
	// read the data from.
	char flag = 0x01;
	//channel_scal* bottmos[2];
	//bottom[0] = bottom0;
	//bottom[1] = bottom1;

	for (char i = 0; i < config_size; i++) {
		memwr_configuration config = read_channel_intel(memwr_configuration_channel);

		int conv_x = config.conv_x;
		int conv_y = config.conv_y;
		int conv_z = config.conv_z;
		int conv_t = config.conv_t;
		int weight_w = config.weight_w;
		bool is_sum_layer_on = config.is_sum_layer_on;
		
		int write_index = 0;

		// We assume conv_z is divisble by LANE_NUM
		uint num_plates = conv_y * (conv_z/LANE_NUM) * conv_t * ((conv_x-1)/W_INV_VEC + 1);
		
		//if (i >= 13)
		//	printf ("[FPGA][memWrite][%d] conv_x=%d, conv_y=%d, conv_z=%d, weight_w=%d, num_plates=%d\n", i, conv_x, conv_y, conv_z, weight_w, num_plates);

		for (uint plate = 0; plate < num_plates; plate++) {
			inv_rows inv;

			inv = read_channel_intel(winograd_inv_transform_channels);

			//if (i >= 13)
			//	printf ("[FPGA][memWrite][%d] plate=%d\n", i, plate);
			
			//if (i >= 13)
			//	printf ("[FPGA][memWrite][%d] start writing to memory!\n", i);
			#pragma unroll
			for (char w = 0; w < W_INV_VEC; w++) {
				// !@^ start of sum_layer
				if(is_sum_layer_on) {
					channel_scal partial_result;
					if (flag == 0x00)
						partial_result = bottom0[write_index+w];
					else
						partial_result = bottom1[write_index+w];
					#pragma unroll
					for(int l = 0; l < LANE_NUM; l++) {
						inv.cols[w].lane[l] += partial_result.lane[l];
					}
				}
				// !@$ end of sum_layer

				if (flag == 0x00)
					bottom0[write_index+w] = inv.cols[w];
				else
					bottom1[write_index+w] = inv.cols[w];
			}

			//if (i >= 12)
			//	printf ("[FPGA][memWrite][%d] finished writing to memory!\n", i);
			write_index += W_INV_VEC;
		}
		flag = (~flag) & 0x01;	
	}
}
