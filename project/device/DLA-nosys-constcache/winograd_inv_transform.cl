// Winograd inverse transformer 
__kernel
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(1)))

void winogradInvTransform()
{

	// Inverse transformation runs forever to convert back the
	// computed data
	while (1) {

		channel_cols input = read_channel_intel(chain_output_channels[LANE_NUM]);

		// inverse transform the input
		// The size of the final output is W_VEC * LANE_NUM
		// After inverse transform, the size should be (W_VEC-weight_w+1) * LANE_NUM

		inv_rows inv;

		#pragma unroll
		for (char i = 0; i < LANE_NUM; i++) {
			#pragma unroll
			for (char j = 0; j < W_INV_VEC; j++) {
				inv.cols[j].lane[i] = input.cols[i].w_data[j];
			}
		}

		write_channel_intel(winograd_inv_transform_channels, inv);
	}

}
