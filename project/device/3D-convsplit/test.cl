#pragma OPENCL EXTENSION cl_intel_channels : enable

#define N 4
channel int chain_channels[N+1];
channel int chain2_channels[N+1];

__attribute__((max_global_work_dim(0)))
__kernel void reader(global int *data_in, int size) {
	for (int i = 0; i < size; ++i) {
		write_channel_intel(chain_channels[0], data_in[i]);
		write_channel_intel(chain2_channels[0], data_in[i]);
	}
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))	
__attribute__((num_compute_units(N)))					
__kernel void plusOne() {
	int compute_id = get_compute_id(0);
	int input = read_channel_intel(chain_channels[compute_id]);
	write_channel_intel(chain_channels[compute_id+1], input + 1);
	int input2 = read_channel_intel(chain2_channels[compute_id]);
	write_channel_intel(chain2_channels[compute_id+1], input2 + 1);

}

__attribute__((max_global_work_dim(0)))
__kernel void writer(global int *data_out, int size) {
	for (int i = 0; i < size; ++i) {
		data_out[i] = read_channel_intel(chain_channels[N]);;
		data_out[i+1] = read_channel_intel(chain2_channels[N]);;
	}
}
