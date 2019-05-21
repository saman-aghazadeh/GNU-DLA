__kernel
__attribute__((task))
void maxPool(
			// Params Ports
			uint  input_num,
			uchar line_size,  // line_size should be no larger than POOL_LBUF_DEPTH
			uchar pool_size,  // by now, only pooling size no larger than 3
			uchar pool_stride
			
			)
{
	channel_scal conv_ch_out;
	channel_scal pool_final;

	//DPTYPE line_buf_0[LANE_NUM][POOL_LBUF_DEPTH];
	//DPTYPE line_buf_1[LANE_NUM][POOL_LBUF_DEPTH];
	DPTYPE line_buf_0[POOL_LBUF_DEPTH][LANE_NUM];
	DPTYPE line_buf_1[POOL_LBUF_DEPTH][LANE_NUM];

	uchar  line_buf_ptr;
	uchar  col_pool_cnt;
	uchar  row_pool_cnt;
	uchar  row_cnt;
	DPTYPE row_pool_reg[LANE_NUM];
	DPTYPE col_pool_reg[LANE_NUM];
	//DPTYPE pool_reg[LANE_NUM][POOL_MAX_SIZE];
	DPTYPE pool_reg[POOL_MAX_SIZE][LANE_NUM];
	
	// Each iteration consumes one output from convolution kernel
	// and then Pooling is performed in column and row directions
	line_buf_ptr = 0;
	row_pool_cnt = 0;
	col_pool_cnt = 0;
	for(unsigned int k=0; k<input_num; k++){

#ifndef EMULATE
		conv_ch_out = read_channel_intel(conv_ch);
#else
		conv_ch_out = read_channel_intel(conv_ch_read);
#endif	
		// Two line buffer to form the 3x3 pooling window
		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			if(pool_size==3)
				row_pool_reg[ll] = pool_max(line_buf_1[line_buf_ptr][ll], line_buf_0[line_buf_ptr][ll]);
			else // pool_size==2
				row_pool_reg[ll] = line_buf_0[line_buf_ptr][ll];
			
			pool_reg[0][ll] = pool_max(row_pool_reg[ll], conv_ch_out.lane[ll]);
			
			if(pool_size==3)
				col_pool_reg[ll] = pool_max(pool_reg[1][ll], pool_reg[2][ll]);
			else //pool_size==2
				col_pool_reg[ll] = pool_reg[1][ll];

			pool_final.lane[ll] = pool_max(col_pool_reg[ll], pool_reg[0][ll]);

			line_buf_1[line_buf_ptr][ll] = line_buf_0[line_buf_ptr][ll];
			line_buf_0[line_buf_ptr][ll] = conv_ch_out.lane[ll];

			#pragma unroll
			for(unsigned char p=POOL_MAX_SIZE-1; p>0; p--){
				pool_reg[p][ll]=pool_reg[p-1][ll];
			}
		}
		
		#ifdef DEBUG_POOL
		printf("Maxpool input_num=%d, line_buf_ptr=%d, row_pool_cnt=%d, col_pool_cnt=%d\n", k, line_buf_ptr, row_pool_cnt, col_pool_cnt);
		printf("        row_cnt=%d\n", row_cnt);
		#endif
		
		// Generates pooling pipeline register wr/rd pointer
		if(row_pool_cnt==(pool_size-1)){

			if(col_pool_cnt==(pool_size-1)){
				write_channel_intel(pool_ch, pool_final);
				#ifdef DEBUG_POOL
				printf("        reg0=%f, reg1=%f, reg2=%f, max=%f\n", (float)pool_reg[0][0], (float)pool_reg[1][0], (float)pool_reg[2][0], (float)pool_final.lane[0]);
				#endif

				col_pool_cnt = (pool_size-pool_stride);
			}
			else
				col_pool_cnt = col_pool_cnt + 1;
		}
		else
			col_pool_cnt = 0;

		// Generates line buffer wr/rd pointer
		if(line_buf_ptr==(line_size-1)){
			line_buf_ptr = 0;

			// Row counters for recognize frames
			if(row_cnt == (line_size-1)) // assuming row_num = line_size, i.e. rectangular frame
				row_cnt = 0;
			else
				row_cnt = row_cnt + 1;

			// Pooling window slide counter for rows
			if(row_cnt == 0)
				row_pool_cnt = 0;
			else if(row_pool_cnt==(pool_size-1))
				row_pool_cnt = (pool_size-pool_stride);
			else
				row_pool_cnt = row_pool_cnt + 1;
		}
		else{
			line_buf_ptr = line_buf_ptr + 1;
		}

	}
}