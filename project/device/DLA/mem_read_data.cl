// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))

void memReadData(
		// Number of layers involved
		char config_size,
		// Data Ports
		__global lane_data	*restrict bottom0,
		__global lane_data 	*restrict bottom1)

{

	for (char i = 0; i < config_size; i++) {

		// Reading the configuration for the specific layer
		memrd_configuration config = read_channel_intel(memrd_configuration_channel);

		// Now we have to read W_VEC * VEC_SIZE elements, that is going to be used 
		// for each iteration of the convolution. We first read through the rows,
		// and the we go through the channels.
		// You have to consider that for each output, what layers are required.
		

	}


}