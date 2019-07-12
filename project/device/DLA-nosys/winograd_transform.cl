// Winograd transformer 
__kernel
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(1)))

void winogradTransform()
{

	// Transformation runs forever to convert the incoming data
	// to the winograd counterpart
	// printf ("[FPGA][winogradTransform] Started!\n");

	while (1) {

		// printf ("[FPGA][winogradTransform] waiting to receive a data!\n");

		lane_cols input = read_channel_intel(winograd_transform_channels);
		lane_cols wd, transformed;
		#pragma unroll
		for(char c = 0; c < VEC_SIZE; c++) {
			const DPTYPE const_0_25 = 0.25f;

			// Compute wd0 := d0 - d6
			wd.cols[0].data[c] = input.cols[0].data[c] - input.cols[6].data[c];
			const DPTYPE d4_sub_d2 = input.cols[4].data[c] - input.cols[2].data[c];
			// Compute wd7 := d7 - d1
			wd.cols[7].data[c] = input.cols[7].data[c] - input.cols[1].data[c];
			const DPTYPE d3_sub_d5 = input.cols[3].data[c] - input.cols[5].data[c];
			// Compute wd1 := d2 + d6
			wd.cols[1].data[c] = input.cols[2].data[c] + input.cols[6].data[c];
			// Compute wd2 := d1 + d5
			wd.cols[2].data[c] = input.cols[1].data[c] + input.cols[5].data[c];
			// Compute wd4 := d5 + 0.25 * d1
			wd.cols[4].data[c] = input.cols[5].data[c] + const_0_25 * input.cols[1].data[c];
			// Compute wd5 := d6 - 5.0 * d4
			wd.cols[5].data[c] = input.cols[6].data[c] - 5 * input.cols[4].data[c];
			// Compute wd3 := d6 + 0.25 * d2
			wd.cols[3].data[c] = input.cols[6].data[c] + const_0_25 * input.cols[2].data[c];
			// Compute wd6 := d1 + 0.25 * d5
			wd.cols[6].data[c] = input.cols[1].data[c] + const_0_25 * input.cols[5].data[c];

			const DPTYPE const_5_25 = 5.25f;
			// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
			wd.cols[0].data[c] += const_5_25 * d4_sub_d2;
			// Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
			wd.cols[7].data[c] += const_5_25 * d3_sub_d5;

			const DPTYPE const_4_25 = 4.25f;
			// Compute
			//   wd1 := (d6 + d2) - 4.25 * d4
			//   wd2 := (d1 + d5) - 4.25 * d3
			wd.cols[1].data[c] -= const_4_25 * input.cols[4].data[c];
			wd.cols[2].data[c] -= const_4_25 * input.cols[3].data[c];

			const DPTYPE const_1_25 = 1.25f;
			// Compute
			//   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
			//   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
			//   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
			//   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
			wd.cols[3].data[c] -= const_1_25 * input.cols[4].data[c];
			const DPTYPE d3_times_1_25 = input.cols[3].data[c] * const_1_25;
			wd.cols[5].data[c] += 4 * input.cols[2].data[c];
			wd.cols[4].data[c] -= d3_times_1_25;
			wd.cols[6].data[c] -= d3_times_1_25;

			const DPTYPE const_2 = 2;
			wd.cols[4].data[c] *= const_2;
			wd.cols[6].data[c] *= const_2;

			transformed.cols[0].data[c] = wd.cols[0].data[c];
			transformed.cols[1].data[c] = wd.cols[1].data[c] + wd.cols[2].data[c];
			transformed.cols[2].data[c] = wd.cols[1].data[c] - wd.cols[2].data[c];
			transformed.cols[3].data[c] = wd.cols[3].data[c] + wd.cols[4].data[c];
			transformed.cols[4].data[c] = wd.cols[3].data[c] - wd.cols[4].data[c];
			transformed.cols[5].data[c] = wd.cols[5].data[c] + wd.cols[6].data[c];
			transformed.cols[6].data[c] = wd.cols[5].data[c] - wd.cols[6].data[c];
			transformed.cols[7].data[c] = wd.cols[7].data[c];
		}

		// Later we have to do something here

		// printf ("[FPGA][winogradTransform] sending the data to the first data channel!\n");

		write_channel_intel(chain_data_channels[0], transformed);

	}

}
