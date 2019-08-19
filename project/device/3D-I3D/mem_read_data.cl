// Deserializing, if necessary
__kernel
__attribute__((max_global_work_dim(0)))
void deser (
		char device_number,
		char deser_data,
		__global ulong4	*restrict bottom)
{

	printf ("[FPGA][Deser][DEV%d] start of deserializer\n", device_number);

	// if we have to receive something from the previous FPGA,
	// we have to wait and read it.
	if (deser_data) {
		memrd_data_deser_configuration config = read_channel_intel(memrd_data_deser_configuration_channel);
		
		int data_w = config.data_w;
		int data_h = config.data_h;
		int data_t = config.data_t;
		int weight_n = config.weight_n;

		int total_size = data_w * data_h * data_t * weight_n;
		total_size = ((total_size + 31)) / 32;

		printf ("[FPGA][Deser][DEV%d] deserilizing with data_w=%d, data_h=%d, data_t=%d, weight_n=%d, total_size=%d\n",
                        device_number,
                        config.data_w,
                        config.data_h,
			config.data_t,
                        config.weight_n,
			total_size);

		for (int i = 0; i < total_size; i++) {
			//printf ("[FPGA][Deser][DEV%d] deserializer receiving some data\n", device_number);
			ulong4 buf;
			buf = read_channel_intel(deser_ch);
			bottom[i] = buf;
		}
	}
	
	printf ("[FPGA][Deser][DEV%d] end of deserializer\n", device_number);	


}

// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))

void memReadData(
		char device_number,
		// Number of layers involved
		char config_size,
		char start_buffer,
		// Data Ports
		__global lane_data	*restrict bottom0,
		__global lane_data 	*restrict bottom1)

{

	// This flag specifies which global data part we should
	// read the data from.
	char flag = start_buffer;

	for (char i = 0; i < config_size; i++) {

		// Reading the configuration for the specific layer
		memrd_data_configuration config = read_channel_intel(memrd_data_configuration_channel);
		int layer_type = config.layer_type;
		int data_w = config.data_w;
		int data_h = config.data_h;
		int data_t = config.data_t;
		int weight_m = config.weight_m;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		int weight_n = config.weight_n;
		int weight_t = config.weight_t;
		int conv_padding = config.conv_padding;
		int conv_stride = config.conv_stride;
		int data_w_with_padding = data_w + 2 * conv_padding;
		int data_h_with_padding = data_h + 2 * conv_padding;
		int data_t_with_padding = data_t + 2 * conv_padding;

		//if (i >= 13)
			// printf ("[FPGA][memReadData][%d] layer_type=%d, data_w=%d, data_h=%d, data_t=%d, weight_m=%d, weight_h=%d, weight_w=%d, weight_n=%d, weight_t=%d, conv_padding=%d, conv_stride=%d, data_w_with_padding=%d, data_h_with_padding=%d\n", i, layer_type, data_w, data_h, data_t, weight_m, weight_h, weight_w, weight_n, weight_t, conv_padding, conv_stride, data_w_with_padding, data_h_with_padding);

		// It may seems strange, but it's memReadData responsibility, to let the 
		// PE knows that it has to load a new set of weights.

		// TODO: We assume for now that weight_m is divisble by LANE_NUM
		int out_channel_iter = weight_m / LANE_NUM;
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
			ushort brick_idx_t = 0;
			uint num_bricks = 0;
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

				if (((brick_idx_y+conv_stride > data_h_with_padding-weight_h)) && (brick_idx_x + (W_VEC+1) > data_w_with_padding)) {
					brick_idx_y = 0;
					brick_idx_t+=conv_stride;
				} else if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
					// if(brick_idx_y ==0 && brick_idx_t ==0)
					// 	printf("brick_idx_x: %d\n", brick_idx_x);
					brick_idx_y+=conv_stride;
				}

				if (brick_idx_x + (W_VEC+1) > data_w_with_padding) {
					brick_idx_x = 0;
				} else {
					brick_idx_x += (W_VEC-weight_w+1);
				}

				num_bricks++;
			}
	// printf("[FPGA][memReadData][%d] num_bricks: %d\tnum_plates: %d\n", i, num_bricks, weight_h * (weight_n/VEC_SIZE) * weight_t);
		}

		// Now we are swapping to the alternative buffer, which contains
		// the data for the next layer
		flag = (~flag) & 0x01;
	}

}
