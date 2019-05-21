__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memReadWeight(
			// Params ports
			ushort	weight_dim4_div_lane,
			uchar	conv_x,
			uchar 	conv_y,
			uint 	weight_dim1x2x3,
			__global channel_vec	*restrict weights)

{

	uint conv_xxy = conv_x * conv_y;

	#pragma max_concurrency 1
	for (ushort i = 0; i < weight_dim4_div_lane; i++) {
		channel_vec weight_buffer[WEIGHT_BUF_SIZE/8];
		for (uint p = 0; p < weight_dim1x2x3/VEC_SIZE; p+=4) {
			weight_buffer[p  ] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p  ];
			weight_buffer[p+1] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+1];	
			weight_buffer[p+2] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+2];
			weight_buffer[p+3] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p+3];
		}
		for (uint j = 0; j < conv_xxy; j++) {
			// channel_vec weight_buffer[WEIGHT_BUF_SIZE];
			for (uint p = 0; p < weight_dim1x2x3/VEC_SIZE; p++) {
				channel_vec weight_ch_vec;
				// if ((i & j) == 0) {
				// 	weight_buffer[p] = weights[i*(weight_dim1x2x3/VEC_SIZE) + p];
				// }
				weight_ch_vec = weight_buffer[p];
				write_channel_intel(weight_ch, weight_ch_vec);
			}
		}
	}	 

}