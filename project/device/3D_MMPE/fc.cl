#define FC_WEIGHT_BUF_SIZE 392
__kernel
__attribute__((max_global_work_dim(0)))
void fc_memRead(
			// Number of layers involved
			char config_size,
			// Params ports
			__global lane_cols	*restrict fc_bottom0,
			__global lane_cols	*restrict fc_bottom1,
			__global lane_cols	*restrict weights,
			__global DPTYPE	*restrict biases)

{
	// printf ("[FPGA][memReadWeight] Number of layers is %d\n", config_size);

	uint layer_offset = 0;
	uint offset = 0;
	char flag = 0x01;

	for (char i = 0; i < config_size; i++) {

		memrd_weight_configuration config = read_channel_intel(fc_memrd_configuration_channel);

		int weight_m = config.weight_m;
		int weight_n = config.weight_n;
		int weight_h = config.weight_h;
		int weight_w = config.weight_w;
		int weight_t = config.weight_t;
		ushort num_plates = weight_t * weight_h * (weight_n/VEC_SIZE);

		//if (i >= 13)
		// 	printf ("[FPGA][memReadWeight][%d] weight_m=%d, weight_n=%d, weight_h=%d, weight_w=%d, num_plates=%d\n", i, weight_m, weight_n, weight_h, weight_w, num_plates);

		flag = (~flag) & 0x01;	

		for (ushort plate = 0; plate < num_plates; plate++) {
			//if (i >= 13)
			//	printf ("[FPGA][memReadWeight][%d] Processing plate %d\n", i, plate);
			lane_cols cur_plate;
			if (flag == 0x00) 
				cur_plate = fc_bottom0[plate];
			else
				cur_plate = fc_bottom1[plate];
			write_channel_intel(fc_data_channel, cur_plate);
		}

		// We assume weight_m is divisible by LANE_NUM
		for (ushort j = 0; j < weight_m; j++) {
			
			//if (i >= 13)
			// 	printf ("[FPGA][memReadWeight][%d] Processing output channel %d\n", i, j*LANE_NUM);

			DPTYPE bias_buffer;
			// Reading LANE_NUM of biases and send them to their 
			// respective PE
			// bias_buffer = biases[j];

			// write_channel_intel(fc_bias_channel, bias_buffer);
			for (ushort plate = 0; plate < num_plates; plate++) {
				//if (i >= 13)
				//	printf ("[FPGA][memReadWeight][%d] Processing plate %d\n", i, plate);
				lane_cols cur_plate = weights[offset];
				offset += 1;
				write_channel_intel(fc_weight_channel, cur_plate);
			}
		}
	}
}


// Let's kill Intel DLA. If you don't share your code with me, I'll write it from
// the scratch.

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fc_PE() {

	// We assume the size of the WEIGHT_BUF_SIZE should be at least 
	// weight_height * weight_dim3 / VEC_SIZE, which we should pick 
	// among the biggest ones.
	__local lane_cols weight_buffer[FC_WEIGHT_BUF_SIZE];

	DPTYPE bias;
	// int id = get_compute_id(0);

	int iter = 0; 

	// Every PE is working all the time. It should loop forver to compute new outputs,
	// and also receive new weights for the next set of output features.
	// This while loop can be considered for computation of layer, one after another one.
	// As a result, each iteration of this while loop maps into processing a specific layer
	while (true) {

		// Reading the instruction required for the number of multiplication for one output
		instruction inst = read_channel_intel(fc_instruction_channel);

		// Bypassing the instruction to the next PE
		// write_channel_intel(chain_instruction_channels[$peid+1$], inst);

		// conv_loop_cnt realizes how many iteration is required to
		// get output for a row of output for this specific output
		// feature. For now I assume it should be
		// weight_height * weights_dim3/VEC_SIZE;
		uint conv_loop_cnt = inst.conv_loop_cnt;

		int frac_w = inst.frac_w;
		int frac_din = inst.frac_din;
		int frac_dout = inst.frac_dout;
		int out_ch_per_pe = inst.out_ch_per_pe;
		int num_bricks = inst.num_bricks;

		// Number of weight vectors that we are going to read
		// it again should be equal to weight_height * weights_dim3/VEC_SIZE.
		// 
		int num_weight_plates = inst.num_weight_plates;

		// printf ("[FPGA][PE$peid$][%d] frac_w=%d, frac_din=%d, frac_dout=%d, out_ch_per_pe=%d, num_bricks=%d, num_weight_plates=%d\n", iter, frac_w, frac_din, frac_dout, out_ch_per_pe, num_bricks, num_weight_plates);
		
		int out_ch = 0;

		for (int i = 0; i < num_weight_plates; i++) {
			// Case 1:
			// weight_lane_cols temp_weight = read_channel_intel(chain_weight_channels[id]);
			// write_channel_intel(chain_weight_channels[id+1], temp_weight);
			// weight_buffer[i] = temp_weight.weight[id];
			weight_buffer[i] = read_channel_intel(fc_data_channel);
		}
		// All the work that should be done in this layer
		while (out_ch < out_ch_per_pe) {

			// printf ("[FPGA][PE$peid$][%d] handling out channel %d\n", iter, out_ch);

			// We have to load the weights into the weight_buffer. weights
			// are loaded through the chain_weight
			// Case 1:
			// bias_DPTYPE bias_buffer= read_channel_intel(chain_bias_channels[id]);
			// write_channel_intel(chain_bias_channels[id+1], bias_buffer);	
			// bias = bias_buffer.bias[id];
					
			// Case 2:
			// bias = read_channel_intel(fc_bias_channel);
			
			
			for (int brick = 0; brick < num_bricks; brick++) {
				w_data accumulation;
				//__local w_data acc_sign_exten;
				//__local w_data acc_with_rnd_bit;
				//__local w_data acc_sum_bias;
			
				// printf ("[FPGA][PE$peid$][%d] Handling brick %d\n", iter, brick);	

				// Now it's time to read the inputs and do the calculation
				for (uint i = 0; i < conv_loop_cnt; i++) {
					// Reading data incoming feature data from the incoming input
					// printf ("[FPGA][PE$peid$][%d] Waiting to read something!\n", iter);
					lane_cols feature = read_channel_intel(fc_weight_channel);

					// Bypassing the data to next PE
					// printf ("[FPGA][PE$peid$][%d] Passing the feature to the next PE!\n", iter);
					// write_channel_intel(chain_data_channels[$peid+1$], feature);

					#pragma unroll
					for (char w = 0; w < W_VEC; w++) {
						accumulation.w_data[w] = 
							(accumulation.w_data[w]) + mac(feature.cols[w], weight_buffer[i].cols[w]);
					}
				}
				/*

				// Not sure why we have to do all these
				#pragma unroll
				for (unsigned i = 0; i < W_VEC; i++) {
					if (accumulation.w_data[i] > 0)
						acc_sign_exten.w_data[i] = 0x00;
					else
						acc_sign_exten.w_data[i] = ~(0xFFFFFFFF >> (frac_w+frac_din-frac_dout-1));

					acc_with_rnd_bit.w_data[i] = (acc_sign_exten.w_data[i] | (accumulation.w_data[i] >> (frac_w+frac_din-frac_dout-1))) + 0x01;

					// This part should be fixed
					if (acc_with_rnd_bit.w_data[i] >= 256)
						acc_sum_bias.w_data[i] = MASK9B & 0xFF;
					else if (acc_with_rnd_bit.w_data[i] < -256)
						acc_sum_bias.w_data[i] = MASK9B & 0x100;
					else
						acc_sum_bias.w_data[i] = (MASK9B & acc_with_rnd_bit.w_data[i]) + (bias>>(frac_w+frac_din-frac_dout-1)) + 0x01;

					accumulation.w_data[i] = MASK8B & (acc_sum_bias.w_data[i] >> 0x01);

					
				}
				*/
				// After an array of output is generated, we will bypass that output to the next layer.
				// It actually combines it's own output with the previous layer output, and then pass 
				// it on

				write_channel_intel(fc_output_channel, accumulation);
				// printf ("[FPGA][PE$peid$][%d] written somethng to the output channel!\n", iter);
			}

			out_ch++;
		}
		iter++;
	}
}

// Store Data to Global Memory
__kernel
__attribute__((max_global_work_dim(0)))
void fc_memWrite(
				char config_size,
				// Params Ports
				// Data Ports
                __global w_data *restrict fc_bottom0,
                __global w_data *restrict fc_bottom1
				)
{

	// This flag specified which global data part we should
	// read the data from.
	char flag = 0x00;
	//channel_scal* bottmos[2];
	//bottom[0] = bottom0;
	//bottom[1] = bottom1;

	for (char i = 0; i < config_size; i++) {
		memwr_configuration config = read_channel_intel(fc_memwr_configuration_channel);

		int conv_x = config.conv_x;
		int conv_y = config.conv_y;
		int conv_z = config.conv_z;
		int conv_t = config.conv_t;
		int weight_w = config.weight_w;
		
		int write_index = 0;

		// We assume conv_z is divisble by LANE_NUM
		uint num_plates = conv_y * conv_z * conv_t * conv_x;
		
		//if (i >= 13)
		//	printf ("[FPGA][memWrite][%d] conv_x=%d, conv_y=%d, conv_z=%d, weight_w=%d, num_plates=%d\n", i, conv_x, conv_y, conv_z, weight_w, num_plates);

		flag = (~flag) & 0x01;	
		for (uint plate = 0; plate < num_plates; plate++) {
			w_data output;

			output = read_channel_intel(fc_output_channel);

			//if (i >= 13)
			//	printf ("[FPGA][memWrite][%d] plate=%d\n", i, plate);
			
			//if (i >= 13)
			//	printf ("[FPGA][memWrite][%d] start writing to memory!\n", i);
			if (flag == 0x00)
				fc_bottom0[write_index] = output;
			else
				fc_bottom1[write_index] = output;
			
			//if (i >= 12)
			//	printf ("[FPGA][memWrite][%d] finished writing to memory!\n", i);
			write_index ++;
		}
	}
}
