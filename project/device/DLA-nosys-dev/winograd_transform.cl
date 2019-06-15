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

		// Later we have to do something here

		// printf ("[FPGA][winogradTransform] sending the data to the first data channel!\n");

		write_channel_intel(chain_data_channels[0], input);

	}

}
