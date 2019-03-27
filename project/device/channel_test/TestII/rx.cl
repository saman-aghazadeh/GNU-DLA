/*
 * AUTHOR: Saman Biookaghazadeh @ Arizona State University
 * DATE:   March 22, 2019
 *
 * Receiver Template.
 */

// Channel definition
// we consider channel 0, and we send data through it.
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel ulong4 sch_in0 __attribute__((depth(4))) __attribute__((io("kernel_input_ch0")));

__kernel void receiver () {

        //for (int i = 0; i < 1000; i++) {
                ulong4 rx_data = read_channel_intel(sch_in0);

		printf ("A: %d, B: %d, C: %d, D: %d\n", rx_data.s0, rx_data.s1, rx_data.s2, rx_data.s3);
        //}

}
