// Store Data to Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
				// Params Ports
				uchar  out_dim1,
				uchar  out_dim2,
				ushort out_dim3,
				ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
				uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
				uchar  batch_indx_dim1,
				uchar  batch_indx_dim2,
				uchar  bypass,
				uchar  padd_offset,
				// Data Ports
                __global DPTYPE *restrict top
				)
{
	uchar  global_x = get_global_id(0); // max value 256
	uchar  global_y = get_global_id(1); // max value 256
	ushort global_z = get_global_id(2); // max value 4096
	uchar  local_x 	= get_local_id(0); // max value 256
	uchar  local_y 	= get_local_id(1); // max value 256
	uchar  local_z 	= get_local_id(2); // max value 256

	uchar  index_z_item; // max value 256
	ushort index_z_group;// max value 4096

	channel_scal   output;
	__local DPTYPE buffer[LANE_NUM];

	if(local_z==0){
		if((bypass&0x01)==0x01)
			output = read_channel_intel(bypass_ch);
		else
			output = read_channel_intel(pool_ch);
		#pragma unroll
		for(uchar ll=0; ll<LANE_NUM; ll++){
			buffer[ll]=output.lane[ll];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	// fetch data from local buffer and write back to DDR
	index_z_group = (global_z-padd_offset)/VEC_SIZE;
	index_z_item  = (global_z-padd_offset)%VEC_SIZE;

	if((global_z-padd_offset)<out_dim3 && (global_z>=padd_offset)){

		top[index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];

		#ifdef DEBUG_MEMWR
		//if((global_z-padd_offset) == 0){
			//for(unsigned char ll=0; ll<LANE_NUM; ll++){
			printf("MemWr results= %f (x=%d, y=%d, z=%d, ll=%d)\n", (float)output.lane[0], global_x, global_y, global_z, 0);
			//}
		//	}
		#endif

	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

}