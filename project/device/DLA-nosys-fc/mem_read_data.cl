// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))

void memReadData(
		// Number of layers involved
		char config_size,
		// Data Ports
		__global lane_data	*restrict bottom0,
		__global lane_data 	*restrict bottom1)

{

	// This flag specifies which global data part we should
	// read the data from.
	char flag = 0x00;

	for (char i = 0; i < config_size; i++) {

		// Reading the configuration for the specific layer
		memrd_data_configuration config = read_channel_intel(memrd_data_configuration_channel);
		int layer_type = config.layer_type;
		int data_w = config.data_w;
		int data_h = config.data_h;
		int weight_m = config.weight_m;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		int weight_n = config.weight_n;
		int conv_padding = config.conv_padding;

		int data_w_with_padding = data_w + 2 * conv_padding;
		int data_h_with_padding = data_h + 2 * conv_padding;

		//if (i >= 13)
		//	printf ("[FPGA][memReadData][%d] layer_type=%d, data_w=%d, data_h=%d, weight_m=%d, weight_h=%d, weight_w=%d, weight_n=%d, conv_padding=%d, data_w_with_padding=%d, data_h_with_padding=%d\n", i, layer_type, data_w, data_h, weight_m, weight_h, weight_w, weight_n, conv_padding, data_w_with_padding, data_h_with_padding);

		// It may seems strange, but it's memReadData responsibility, to let the 
		// PE knows that it has to load a new set of weights.

		// TODO: We assume for now that weight_m is divisble by LANE_NUM
		int out_channel_iter = (layer_type == 0) ? weight_m / LANE_NUM : 1;
		//if (i >= 13)
		//	printf ("[FPGA][memReadData][%d] out_channel_iter is %d\n", i, out_channel_iter);	

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
			uint num_bricks = 0;
			while (brick_idx_y != data_h_with_padding-weight_h+1) {

				//if (i >= 13)
				//	printf ("[FPGA][memReadData][%d] Processing a new brick with brick_idx_x=%d and brick_idx_y=%d and id=%d\n", i, brick_idx_x, brick_idx_y, num_bricks);

				// These indexes determines where are we in the 
				// feature map.
				char feature_idx_x = brick_idx_x;
				char feature_idx_y = brick_idx_y;
				char feature_idx_z = 0;

				// TODO: Here assume weight_n is divisible by VEC_SIZE
				short num_plates = weight_h * (weight_n/VEC_SIZE);

				//if (i >= 13)
				//	printf ("[FPGA][memReadData][%d] num_plates is %d\n", i, num_plates);

				for (short plate = 0; plate < num_plates; plate++) {
					//if (i >= 12)
						//printf ("[FPGA][memReadData] Processing plate=%d\n", plate);						
					lane_cols data_for_convs;

					// We have to read a plate of data, which is of 
					// size W_VEC * VEC_SIZE
					short read_index = 
						feature_idx_z * data_w * data_h +
						(feature_idx_y-conv_padding) * data_w +
						(feature_idx_x-conv_padding);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {

						// if ((feature_idx_x+w >= conv_padding && feature_idx_x+w < data_w + conv_padding)
						//	&&
						//	(feature_idx_y >= conv_padding && feature_idx_y < data_h + conv_padding)) {
							if (flag == 0x00) {
								data_for_convs.cols[w] = bottom0[read_index+w];
							} else {
								data_for_convs.cols[w] = bottom1[read_index+w];
							}
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
					write_channel_intel(winograd_transform_channels, data_for_convs);
					//if (i >= 12)
					//	printf ("[FPGA][memReadData][%d] After sending the data\n", i);

					// Alright data is sent, we have to move on to the next plate.
					// That means, we have to update out feature indexes

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

				if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
					brick_idx_x = 0;
					brick_idx_y++;
				} else {
					brick_idx_x += (W_VEC-weight_w+1);
				}

				num_bricks++;
			}
		}

		// Now we are swapping to the alternative buffer, which contains
		// the data for the next layer
		flag = (~flag) & 0x01;
	}

}
