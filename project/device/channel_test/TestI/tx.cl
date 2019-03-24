/*
 * AUTHOR: Saman Biookaghazadeh @ Arizona State University
 * DATE:   March 22, 2019
 * 
 * Sender Template.
 */

// Channel definition
// we consider channel 0, and we send data through it.
channel ulong4 sch_out0 __attribute__((depth(4))) __attribute__((io("kernel_output_ch0")));

__kernel void sender () {

	for (int i = 0; i < 1000; i++) {
		ulong4 tx_data;
		tx_data.s0 = i;
		tx_data.s1 = i+1;
		tx_data.s2 = i+2;
		tx_data.s3 = i+3;

		write_channel_intel(sch_out0, tx_data);
	}

}
