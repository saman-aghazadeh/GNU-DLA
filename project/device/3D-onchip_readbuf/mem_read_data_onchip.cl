
#include "loop.h"
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

	char bottom_flag = 0x00;
	lane_cols __attribute__((numbanks(16),
							bankwidth(VEC_SIZE))) data_buffer[WIN_BUF_SIZE][2];

	for (char i = 0; i < config_size; i++) {
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
		
		int data_w_with_padding = data_w + 2 * conv_padding;
		int data_h_with_padding = data_h + 2 * conv_padding;
		int data_t_with_padding = data_t + 2 * conv_padding;
		char flag = 0x00;

		int conv_group_size_w = config.conv_group_size_w;
		int conv_group_size_h = config.conv_group_size_h;
		int conv_group_size_t = config.conv_group_size_t;
		int conv_last_group_size_w = conv_group_size_w;//config.conv_last_group_size_w;
		int conv_last_group_size_h = (data_h_with_padding - weight_h + 1) % conv_group_size_h;
		if(conv_last_group_size_h == 0) 
			conv_last_group_size_h = conv_group_size_h;
		int conv_last_group_size_t = (data_t_with_padding - weight_t + 1) % conv_group_size_t;
		if(conv_last_group_size_t == 0) 
			conv_last_group_size_t = conv_group_size_t;
		// int conv_last_group_size_t = (conv_group_size_t%2 == 0) ? 2 : 1; //config.conv_last_group_size_t;

		int num_tiles_w = (data_w_with_padding - weight_w) / conv_group_size_w + 1;
		int num_tiles_h = (data_h_with_padding - weight_h) / conv_group_size_h + 1;
		int num_tiles_t = (data_t_with_padding - weight_t) / conv_group_size_t + 1;
		int num_tiles = num_tiles_w * num_tiles_h * num_tiles_t;
		int tiles = 1 + num_tiles * weight_m / LANE_NUM;

		int conv_tile_size_w = (conv_group_size_w + weight_w -1);
		int conv_tile_size_h = (conv_group_size_h + weight_h -1);
		int conv_tile_size_t = (conv_group_size_t + weight_t -1);
		int conv_last_tile_size_w = (conv_last_group_size_w + weight_w -1);
		int conv_last_tile_size_h = (conv_last_group_size_h + weight_h -1);
		int conv_last_tile_size_t = (conv_last_group_size_t + weight_t -1);

		int next_tile_idx_w = 0, next_tile_idx_h = 0, next_tile_idx_t = 0;
		int tile_size_w = 0, tile_size_h = 0, tile_size_t = 0;
		int next_tile_size_w = conv_tile_size_w, next_tile_size_h = conv_tile_size_h, next_tile_size_t = conv_tile_size_t;
		int next_tile_idx_offset = 0;
		// printf("%d %d %d %d\n", conv_tile_size_h, conv_tile_size_t, next_tile_size_t, conv_tile_size_w);
		int counter = 0;
		int w_vec = (layer_type == 1) ? data_w_with_padding : W_VEC;
		// printf("w_vec: %d\n", w_vec);
		int num_plates = weight_h * (weight_n / VEC_SIZE) * weight_t;
		for(int tile_i = 0; tile_i != tiles; tile_i++) {
			int databuf_read_idx_w = 0;
			int databuf_read_idx_h = 0;
			int databuf_read_idx_c = 0;
			int databuf_read_idx_t = 0;

			int databuf_write_idx_w = 0;
			int databuf_write_idx_h = 0;
			int databuf_write_idx_c = 0;
			int databuf_write_idx_t = 0;
			int conv_group_loop_h = 0;
			int conv_group_loop_t = 0;
			int data_load_loop_limit = next_tile_size_h * (weight_n / VEC_SIZE) * next_tile_size_t;
			int compute_loop_limit = num_plates * (tile_size_w / w_vec) * (tile_size_h - weight_h + 1) * (tile_size_t - weight_t + 1);
			int inner_loop_limit = (compute_loop_limit > data_load_loop_limit) ? compute_loop_limit: data_load_loop_limit;
			// printf("%d %d %d %d\n", num_plates, tile_size_w, tile_size_h, tile_size_t);
			// printf("%d\n", compute_loop_limit);
			for(int inner_loop = 0; inner_loop < inner_loop_limit; inner_loop++) {
				if(inner_loop < data_load_loop_limit) {
					int input_idx = next_tile_idx_offset + databuf_read_idx_t * data_w * data_h * weight_n / VEC_SIZE + 
									  databuf_read_idx_c * data_w * data_h + databuf_read_idx_h * data_w;
					int databuf_read_idx = databuf_read_idx_t * next_tile_size_h * weight_n / VEC_SIZE + 
									  databuf_read_idx_c * next_tile_size_h + databuf_read_idx_h;// + databuf_read_idx_w;
					lane_cols lanecols;
					#pragma unroll
					for(char w = 0; w < W_VEC; w++) {
						if(bottom_flag == 0x00)
							lanecols.cols[w] = bottom0[input_idx + w];
						else
							lanecols.cols[w] = bottom1[input_idx + w];
					}
					data_buffer[databuf_read_idx][flag&0x01] = lanecols;
					loop3(databuf_read_idx_t, (next_tile_size_t-1), databuf_read_idx_c, (weight_n/VEC_SIZE-1),
						 databuf_read_idx_h, (next_tile_size_h-1))
				}
				if(inner_loop < compute_loop_limit){
					lane_cols data_for_convs;
					int databuf_write_idx = (conv_group_loop_t + databuf_write_idx_t)  * tile_size_h * weight_n / VEC_SIZE + 
											databuf_write_idx_c *  tile_size_h +
											(conv_group_loop_h + databuf_write_idx_h) ;
					// #pragma unroll
					// for (char w = 0; w < W_VEC; w++) {
						data_for_convs = data_buffer[databuf_write_idx][(~flag)&0x01];
					// }
					write_channel_intel(winograd_transform_channels, data_for_convs);
					// counter++;
					loop5(conv_group_loop_t, (conv_group_size_t-1), conv_group_loop_h, (conv_group_size_h-1),
							 databuf_write_idx_t, (weight_t-1), databuf_write_idx_c, (weight_n/VEC_SIZE-1), databuf_write_idx_h, (weight_h-1))
				}
			}
			loop3(next_tile_idx_t, (num_tiles_t - 1), next_tile_idx_h, (num_tiles_h - 1), next_tile_idx_w, (num_tiles_w - 1))
			next_tile_idx_offset =	(next_tile_idx_t * conv_group_size_t * data_w * data_h * weight_n / VEC_SIZE) + 
										(next_tile_idx_h * conv_group_size_h * data_w )+ 
										(next_tile_idx_w * conv_group_size_w);
			tile_size_w = next_tile_size_w;
			tile_size_h = next_tile_size_h;
			tile_size_t = next_tile_size_t;
			next_tile_size_w = (next_tile_idx_w == num_tiles_w - 1) ? conv_last_tile_size_w : conv_tile_size_w;
			next_tile_size_h = (next_tile_idx_h == num_tiles_h - 1) ? conv_last_tile_size_h : conv_tile_size_h;
			next_tile_size_t = (next_tile_idx_t == num_tiles_t - 1) ? conv_last_tile_size_t : conv_tile_size_t;
			flag = (~flag) & 0x01;
		}
		bottom_flag = (~bottom_flag) & 0x01;
		// printf("read counter: %d for layer %d\n", counter, i);
	}

}

