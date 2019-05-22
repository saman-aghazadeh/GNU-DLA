// Winograd transformer 
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))

void winogradTransform()
{

	lane_cols input = read_channel_intel(winograd_transform_channels);

	// Later we have to do something here

	write_channel_intel(chain_data_channels[0], input);

}
