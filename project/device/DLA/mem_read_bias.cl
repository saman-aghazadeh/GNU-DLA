__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memReadBias(
			// Params ports
			ushort	weight_dim4_div_lane,	// avoid generating divider
			uchar 	conv_x,
			uchar	conv_y,
			__global channel_scal	*restrict bias)
{	
		

	channel_scal	bias_ch_in;
	ushort conv_xy = conv_x * conv_y;

	for (ushort i = 0; i < weight_dim4_div_lane; i++) {
		bias_ch_in = bias[i];
		for (ushort j = 0; j < conv_xy; j++) {
			write_channel_intel(bias_ch, bias_ch_in);
		}
	}


}