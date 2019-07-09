// Store Data to Global Memory
__kernel
__attribute__((max_global_work_dim(0)))
void memWrite(
				char device_number,
				char config_size,
				char start_buffer,
				// Params Ports
				// Data Ports
                __global channel_scal *restrict bottom0,
                __global channel_scal *restrict bottom1
				)
{

	// This flag specified which global data part we should
	// read the data from.
	char flag = start_buffer;
	//channel_scal* bottmos[2];
	//bottom[0] = bottom0;
	//bottom[1] = bottom1;

	for (char i = 0; i < config_size; i++) {
		memwr_configuration config = read_channel_intel(memwr_configuration_channel);

		int conv_x = config.conv_x;
		int conv_y = config.conv_y;
		int conv_z = config.conv_z;
		int weight_w = config.weight_w;
		
		int write_index = 0;

		// We assume conv_z is divisble by LANE_NUM
		uint num_plates = conv_y * (conv_z/LANE_NUM) * ((conv_x-1)/W_INV_VEC + 1);

		printf ("[FPGA][memWrite][DEV%d][%d] conv_x=%d, conv_y=%d, conv_z=%d, weight_w=%d, num_plates=%d\n", device_number, i, conv_x, conv_y, conv_z, weight_w, num_plates);

		for (uint plate = 0; plate < num_plates; plate++) {
			inv_rows inv;

			inv = read_channel_intel(winograd_inv_transform_channels);

			printf ("[FPGA][memWrite][DEV%d][%d] plate=%d\n", device_number, i, plate);
			
			// printf ("[FPGA][memWrite][%d] start writing to memory!\n", i);
			#pragma unroll
			for (char w = 0; w < W_INV_VEC; w++) {

				if (flag == 0x00)
					bottom0[write_index+w] = inv.cols[w];
				else
					bottom1[write_index+w] = inv.cols[w];
			}
			// printf ("[FPGA][memWrite][%d] finished writing to memory!\n", i);
			write_index += W_INV_VEC;
		}
		flag = (~flag) & 0x01;	
	}
}


// Serializing, if necessary
__kernel
__attribute__((max_global_work_dim(0)))
void ser(
		char device_number,
		char ser_data,
		__global ulong4		*restrict bottom)
{

	// If we have to send something to the next fpga,
	// then we have to send it over the serial channel
	if (ser_data) {
		memrd_data_ser_configuration config = read_channel_intel(memrd_data_ser_configuration_channel);
		
		int nl_data_w = config.nl_data_w;
		int nl_data_h = config.nl_data_h;
		int nl_weight_n = config.nl_weight_n;

		int total_size = nl_data_w * nl_data_h * nl_weight_n;
		total_size = ((total_size + 31) / 32);

                printf ("[FPGA][ser][DEV%d] serializing with data_w=%d, data_h=%d, weight_n=%d\n",
                        device_number,
                        memrd_data_ser_config.nl_data_w,
                        memrd_data_ser_config.nl_data_h,
                        memrd_data_ser_config.nl_weight_n); 

		for (int i = 0; i < total_size; i++) {
			ulong4 buf;
			buf = bottom[i];

			write_channel_intel(ser_ch, buf);
		}
	}	

}
