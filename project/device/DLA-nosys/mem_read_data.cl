// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))

void memReadData(
		// Number of layers involved
		char config_size,
		// Data Ports
		__global volatile lane_data	*restrict bottom0,
		__global volatile lane_data 	*restrict bottom1)

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

		// It may seems strange, but it's memReadData responsibility, to let the 
		// PE knows that it has to load a new set of weights.

		// TODO: We assume for now that weight_m is divisble by LANE_NUM
		char out_channel_iter = weight_m / LANE_NUM;
		
		for (char j = 0; j < out_channel_iter; j++) {

			// We have to read the data brick by brick.
			// Every brick is of size 
			// W_VEC * weight_h * weight_n
			// weight_w has been replaced with W_VEC, since we are
			// going to do winograd convolution :D

			char brick_idx_x = 0;
			char brick_idx_y = 0;

			while (brick_idx_y != data_h_with_padding-weight_h+1) {

				// These indexes determines where are we in the 
				// feature map.
				char feature_idx_x = brick_idx_x;
				char feature_idx_y = brick_idx_y;
				char feature_idx_z = 0;

				// TODO: Here assume weight_n is divisible by VEC_SIZE
				short num_plates = weight_h * (weight_n/VEC_SIZE);

				for (short plate = 0; plate < num_plates; plate++) {
					lane_cols data_for_convs;

					// We have to read a plate of data, which is of 
					// size W_VEC * VEC_SIZE
					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						short read_index = 
							feature_idx_z * data_w * data_h +
							(feature_idx_y-conv_padding) * data_w +
							(feature_idx_x-conv_padding) + 
							w;

						if ((feature_idx_x+w >= conv_padding && feature_idx_x+w < data_w + conv_padding)
							&&
							(feature_idx_y >= conv_padding && feature_idx_y < data_h + conv_padding)) {
							if (flag == 0x00) {
								data_for_convs.cols[w] = bottom0[read_index];
							} else {
								data_for_convs.cols[w] = bottom1[read_index];
							}
						} else {
							#pragma unroll
							for (unsigned char vv = 0; vv < VEC_SIZE; vv++)
								data_for_convs.cols[w].data[vv] = CZERO;
						}
					}

					// DAMN we read the plate. Now it's time for peanut butter jelly.
					// Just kidding! we have to send the data to the first PE. 
					write_channel_intel(winograd_transform_channels, data_for_convs);

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

				if ((brick_idx_y == data_h_with_padding-1) && (brick_idx_x+(W_VEC-weight_w+1) >= data_w_with_padding-1 )){
					brick_idx_y = 0;
				} else if (brick_idx_x + (W_VEC-weight_w+1) >= data_w_with_padding-1) {
					brick_idx_y++;
				}

				if (brick_idx_x + (W_VEC-weight_w+1) >= data_w_with_padding-1) {
					brick_idx_x = 0;
				} else {
					brick_idx_x += (W_VEC-weight_w+1);
				}
			}
		}

		// Now we are swapping to the alternative buffer, which contains
		// the data for the next layer
		flag = (~flag) & 0x01;
	}

}
