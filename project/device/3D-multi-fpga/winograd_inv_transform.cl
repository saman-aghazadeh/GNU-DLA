// Winograd inverse transformer 
__kernel
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(1)))

void winogradInvTransform()
{

	// printf ("[FPGA][winogradInvTransform] Started!\n");

	// Inverse transformation runs forever to convert back the
	// computed data
	while (1) {

		// printf ("[FPGA][winogradInvTransform] Waiting for data\n");
		channel_cols31 input = read_channel_intel(chain_output_channels31);
		// printf ("[FPGA][winogradInvTransform] Dome waiting for data\n");

		// inverse transform the input
		// The size of the final output is W_VEC * LANE_NUM
		// After inverse transform, the size should be (W_VEC-weight_w+1) * LANE_NUM

		inv_rows inv;

		#pragma unroll
		for(char l = 0; l < LANE_NUM; l++) {
			/*
			 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
			 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
			 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
			 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
			 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
			 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
			 */

			const DPTYPE m1_add_m2 = input.cols[l].w_data[1] + input.cols[l].w_data[2];
			const DPTYPE m1_sub_m2 = input.cols[l].w_data[1] - input.cols[l].w_data[2];
			const DPTYPE m3_add_m4 = input.cols[l].w_data[3] + input.cols[l].w_data[4];
			const DPTYPE m3_sub_m4 = input.cols[l].w_data[3] - input.cols[l].w_data[4];
			const DPTYPE m5_add_m6 = input.cols[l].w_data[5] + input.cols[l].w_data[6];
			const DPTYPE m5_sub_m6 = input.cols[l].w_data[5] - input.cols[l].w_data[6];

			DPTYPE s0 = input.cols[l].w_data[0] + m1_add_m2;
			DPTYPE s5 = input.cols[l].w_data[7] + m1_sub_m2;

			const DPTYPE const_16 = 16;
			DPTYPE s1 = m1_sub_m2 + const_16 * m5_sub_m6;
			DPTYPE s4 = m1_add_m2 + const_16 * m3_add_m4;

			const DPTYPE const_8 = 8;
			DPTYPE s2 = m1_add_m2 + const_8 * m5_add_m6;
			DPTYPE s3 = m1_sub_m2 + const_8 * m3_sub_m4;

			const DPTYPE const_32 = 32;
			s0 += const_32 * m5_add_m6;
			s5 += const_32 * m3_sub_m4;

			s0 += m3_add_m4;
			s5 += m5_sub_m6;

			const DPTYPE const_2 = 2;
			s1 += m3_sub_m4 * const_2;
			s4 += m5_add_m6 * const_2;

			const DPTYPE const_4 = 4;
			s2 += m3_add_m4 * const_4;
			s3 += m5_sub_m6 * const_4;

			inv.cols[0].lane[l] = s0;
			inv.cols[1].lane[l] = s1;
			inv.cols[2].lane[l] = s2;
			inv.cols[3].lane[l] = s3;
			inv.cols[4].lane[l] = s4;
			inv.cols[5].lane[l] = s5;
		}

		// #pragma unroll
		// for (char i = 0; i < LANE_NUM; i++) {
		// 	#pragma unroll
		// 	for (char j = 0; j < W_INV_VEC; j++) {
		// 		inv.cols[j].lane[i] = input.cols[i].w_data[j];
		// 	}
		// }

		// printf ("[FPGA][winogradInvTransform] Writing something to the mem_write channel\n");
		write_channel_intel(winograd_inv_transform_channels, inv);
		// printf ("[FPGA][winogradInvTransform] Done writing something to the mem_write channel\n");
	}

}


