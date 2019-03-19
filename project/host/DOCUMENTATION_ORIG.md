## PipeCNN (Original) Host Code Explanation

In this part of the documentation, I would like to go through the important parts of the source code and explain how exactly they work.

### Kernel Names
~~~
	const char *knl_name_memRd = "memRead";
	const char *knl_name_conv  = "coreConv";
	const char *knl_name_Pool  = "maxPool";
	const char *knl_name_memWr = "memWrite";
	const char *knl_name_lrn   = "lrn";
	const char *knl_name_ser   = "lrnSer";
	const char *knl_name_deser = "memReadDeser";
~~~
This specific part specifies the kernels that exist in the code. Need to mention that serializer and deserializer (specified as `lrnSer` and `memReadDeser`) are specified by ourselves. More informations about these kernels can be found in the other readme file in the `device` foolder.

### OpenCL Data Structures
~~~
	cl_uint num_devices = 0;
	cl_platform_id platform_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	scoped_array<cl_device_id> device;
	scoped_array<cl_kernel> knl_memRd;
	scoped_array<cl_kernel> knl_conv;
	scoped_array<cl_kernel> knl_memWr;
	scoped_array<cl_kernel> knl_pool;
	scoped_array<cl_kernel> knl_lrn;
	scoped_array<cl_kernel> knl_ser;
	scoped_array<cl_kernel> knl_deser;
	scoped_array<cl_command_queue> que_memRd;
	scoped_array<cl_command_queue> que_conv;
	scoped_array<cl_command_queue> que_memWr;
	scoped_array<cl_command_queue> que_pool;
	scoped_array<cl_mem> data_buf;
	scoped_array<cl_mem> output_buf;
	scoped_array<cl_mem> weights_buf;
	scoped_array<cl_mem> bias_buf;
	scoped_array<cl_mem> fc_1_buf;
	scoped_array<cl_mem> fc_2_buf;
~~~

In the code we save the number of devices (`num_devices`), the id of the platform (`platform_id`), context and the program and the list of the devices. We save the generate OpenCL kernels for each device. For each of the kernels, we generate one respective `cl_kernel` object for that. Further, we have four specific command queues, known as `que_memRd`, `que_conv`, `que_memWr`, `que_pool`. At the end, we have five specific allocated buffers for each device. They are `data_buf`, `output_buf`, `weights_buf`, `bias_buf`, `fc_1_buf`, and `fc_2_buf`. We will discuss how these buffers are being used later on. 

### Host Data Structures
~~~
	DTYPE *weights;
	DTYPE *image;
	DTYPE *data_init;
	DTYPE *weight_conv[MAX_LAYER_NUM];
	DTYPE *bias_conv[MAX_LAYER_NUM];
	DTYPE *output;
	DTYPE *output_one_item;
	DTYPE *output_reorder;
	DTYPE *golden_ref;
~~~

`weights` holds are the model weight, all at once. 
`image` is supposed to hold the input image (or even the batch of image).
`data_init` I don't know yet!
`weight_conv`, holds the weights for each separate layer. 
`bias_conv` holds the biases for eachseparate layer.
`output` holds the output value of the model.
`output_one_iteam` I don't know yet!
`output_reorder` I don't know yet!
`golden_ref` the reference value.

## Preparation of the Data
~~~
    for(unsigned ii=0; ii<NUM_CONFIG_ITEM; ii++){
        layer_config_original[ll][ii]=layer_config[ll][ii];
    }
~~~
This part of the code backs up the original configuration parameters for all the layers. 

~~~
        if(layer_config[ll][weight_m]%LANE_NUM != 0){
            printf("\nWarnning: layer-%d requires padding zero-value feature maps for give param LANE_NUM=%d\n", ll+1, LANE_NUM);
            layer_config[ll][weight_m] = ceil((float)layer_config[ll][weight_m]/LANE_NUM)*LANE_NUM;
            layer_config[ll][bias_size] = layer_config[ll][weight_m];
            printf("      original num of feature maps is %d, new value is %d\n", layer_config_original[ll][weight_m], layer_config[ll][weight_m]);

            // padding of weight on dim4 is needed
            padding_offset[ll] = layer_config[ll][weight_m] - layer_config_original[ll][weight_m];
            // check if evenly padding on two sides is possible
            if(((layer_config[ll][weight_m]/LANE_NUM)%2!=0) & (layer_config[ll][conv_split]==1)){
                printf("Error: could not perform padding for split mode, weight_m/LANE_NUM must be divisible by 2 !!!\n\n");
                return 1;
            }
            else{ // padding zeros evenly on two sides of dim4
                padding_offset[ll] = padding_offset[ll]/2;
                printf("      padding_offset=%d (layer=%d)\n\n", padding_offset[ll], ll+1);
            }

        }
~~~

the number of output channels (`weight_m`), should be divisible by the `LANE_NUM`, so that that the FPGA can output total of `LANE_NUM` values through the channels. In case that this number is not divisible, then the weights and bias size would get updated. Also, the offset, which is the difference between the old `weight_m` and the new values, will be stored for each layer. And then, this offset will be divided by two, since we would like to have half of the total offset on both sides of the output channel. Usually, thats how the offset works.

For some rest of the few lines of codes, it will does sanity checking to make sure all dimensions of the consecutive layers are compatible.

~~~
    if(layer_config[ll][conv_x]==1){ // when only one group for FC layer
        conv_win_size_dim1  = layer_config[ll][weight_w];
    }
    else{
        conv_win_size_dim1  = layer_config[ll][weight_w]+(CONV_GP_SIZE_X-1)*layer_config[ll][conv_stride];
    }
    conv_win_size_dim2    = layer_config[ll][weight_h];
~~~

In this part, the code sets the window size first and second dimension. I honestly do not understand this part, yet. Will have to take a look into the kernel code to see what is the role of the window. some primary speculations tells me that it may seems like the same windowing method that we have seen in the shift registers. 

~~~
    if(conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/VEC_SIZE > WIN_BUF_SIZE){

        printf("Error: required win_buffer size is %d, configured size is %d \n", conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/VEC_SIZE, WIN_BUF_SIZE);
        return 1;
    }
    // check weight_buffer size
    if(layer_config[ll][weight_w]*layer_config[ll][weight_h]*layer_config[ll][weight_n]/VEC_SIZE > WEIGHT_BUF_SIZE){

        printf("Error: required weight_buffer size is %d, configured size is %d \n", layer_config[ll][weight_w]*layer_config[ll][weight_h]*layer_config[ll][weight_n]/VEC_SIZE, WEIGHT_BUF_SIZE);
        return 1;
    }
~~~
This part of the code checks whether we have enough window and weight buffer size for our model. If the buffer sizes are smaller, then we are in a trouble. One might need to increase the size of these buffers.

~~~
    layer_config[0][weight_n] = ceil((float)layer_config[0][weight_n]/VEC_SIZE)*VEC_SIZE;
    layer_config[0][data_n] = layer_config[0][weight_n];
~~~

The input channels need to be padded as well. The input channel is being vectorized and then processed by the kernels. As a result, the input channels should be divisible by the `VEC_SIZE` value. Rest assured that the next layers channels have already comply with the `VEC_SIZE`, since we already did the sanity check.

~~~
    data_init   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n], DMA_ALIGNMENT);
    memset(data_init, 0, sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]);// fill non-RGB dims with 0
~~~

In this part, we can observe that we allocate the `data_init` buffer, which is the size of the very first layer, without considering the output channels. In fact, it is the size of the image (don't forget that it already includes the padding as well). So, whatever that is padded is being filled with zeros. For example, in RGB images, only three channels are filled with numbers, rest are filled with zero.

~~~
   if(LAYER_NUM>=CONV_NUM)// For last conv and all fc layers, all batch results are read back
        output_size = output_config[output_w]*output_config[output_h]*output_config[output_n]*input_config[batch_size];
    else // For other conv layers, only one item of
        output_size = output_config[output_w]*output_config[output_h]*output_config[output_n];
~~~

In this part we decide the `output_size` value. This value seems to be size of the output of the current layer. If we re not in the last convolution or the fc layer, then the output size is only for one output, but if we are in the last conv or fc layers, then we output the value for the whole batch. (It is still unclear to me why this is the case. In another words, why we need to process the whole batch for the final layers, and how it provides good optimizations.)

~~~
godref_size = output_config[output_w]*output_config[output_h]*output_config[output_n];
~~~

Seems to be the size of the golden reference output. 

~~~
    output          = (DTYPE *)alignedMalloc(sizeof(DTYPE)*output_size, DMA_ALIGNMENT); // vectorized results
    output_one_item = (DTYPE *)alignedMalloc(sizeof(DTYPE)*godref_size, DMA_ALIGNMENT); // one item extracted from batch results
    golden_ref      = (DTYPE *)alignedMalloc(sizeof(DTYPE)*godref_size, DMA_ALIGNMENT);
    output_reorder  = (DTYPE *)alignedMalloc(sizeof(DTYPE)*godref_size, DMA_ALIGNMENT); // reordered results for verifying
~~~

Allocation for the host side output buffers. Everything is aligned, so the FPGA can take advatnage of the DMA.

~~~
    // weights and bias buffers
    for(int j=0; j<LAYER_NUM; j++){

        weight_size = (layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]);
        weight_conv[j] = (DTYPE *)alignedMalloc(sizeof(DTYPE)*weight_size, DMA_ALIGNMENT);
        bias_conv[j]   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*layer_config[j][bias_size], DMA_ALIGNMENT);

        memset(weight_conv[j], 0, sizeof(DTYPE)*weight_size);             // reset all value (include padding value) to zero
        memset(bias_conv[j], 0, sizeof(DTYPE)*layer_config[j][bias_size]);// reset all value (include padding value) to zero

    }
~~~

In this part, for each layer we create he appropriate buffer that holds all the weights and biases. As you can see, the weights include the `width`, `height`, `input channel`s, and `output channel`s. 

In the next three sections of the code, we read the weights, synsets, and the golden reference files. 

At the end, we do reordering of the weights and biases. Next we will discuss how the weight reordering really works.

### Reorder Weights

~~~
    for(unsigned m = 0; m<dim4_original; m++){
        for(unsigned n = 0; n<dim3_original; n++){
            for(unsigned i = 0; i<dim2; i++){
                for(unsigned j = 0; j<dim1; j++){
                            copy_with_padding[(padding_offset*dim1*dim2*dim3) + m*dim1*dim2*dim3 + n*dim1*dim2 + i*dim1 + j]
                                                    = (DTYPE) weights[offset+m*dim1*dim2*dim3_original + n*dim1*dim2 + i*dim1 + j];
                }
            }
        }
    }
~~~

First of all, we have to apply padding on the weights buffer. We aleady have calculated the `padding_offset`. Based on this offset, we store the weights in a new data structure, that contains the padding. As mentioned before, the padding is being done on the output channels, which has highest order amongst all the other dimensions.

~~~
    for(unsigned m = 0; m<(dim4/laneNum); m++){
        for(unsigned n = 0; n<(dim3/vecSize); n++){
            for(unsigned i = 0; i<dim2; i++){
                for(unsigned j = 0; j<dim1; j++){
                    for(unsigned ll = 0; ll<laneNum; ll++){
                        for(unsigned k = 0; k<vecSize; k++){
                            weight_buf[m*dim1*dim2*dim3*laneNum + n*dim1*dim2*vecSize*laneNum + i*dim1*vecSize*laneNum + j*vecSize*laneNum + ll*vecSize + k]
                                                    = (DTYPE) copy_with_padding[(m*laneNum+ll)*dim3*dim2*dim1 + (n*vecSize+k)*dim1*dim2 + i*dim1 + j];
                        }
                    }
                }
            }
        }
    }
~~~

This specific part is responsible for reorder the whole padded dataset. In our kernels, the `LANE_NUM` and the `VEC_SIZE` are being applied on the 3rd and 4th dimensions, which are the input channels and the output channels. As a result, the data organization should be changed respectively. In the old structure, we first write all the data for the output channel. For that specific output channel, the data is organized by the input channel. For every input channel, the is organized by the height, and then for each height we write all the widths. We have to reconsider this organization. We will cluster output channels and input channels into bucks of size `LANE_NUM` and `VEC_SIZE`. Now for the range of output and input channels in that buck, we store the data first ordered by height, then the width. For each data in a `(height, width)` location, we store the ouput and input channels, in that order. As a result, consecutive values are channels values for the same `(height, width)` location.

## Creating the Weight and Biases buffers 
~~~
weight_buf_size = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m];
~~~

The `weight_buf_size` is the total size of the weight of that specific layer, which is equal to the multiplications of the width, the height, the number of input channels, and the number of output channels.

~~~
    weights_buf[i*LAYER_NUM+j] = clCreateBuffer(context,CL_MEM_READ_ONLY,weight_buf_size* sizeof(DTYPE),NULL,&status);
    checkError(status, "Failed to create buffer for weights in layer");

    // Bias buffers for each layer
    bias_buf[i*LAYER_NUM+j] = clCreateBuffer(context,CL_MEM_READ_ONLY,layer_config[j][bias_size] * sizeof(DTYPE),NULL,&status);
    checkError(status, "Failed to create buffer for bias in layer");

~~~

The weight buffers are being created for each layer separately on all the FPGA cards. 


~~~
    data_buf[i*input_config[batch_size]+j] = clCreateBuffer(context, CL_MEM_READ_WRITE, IN_BUF_SIZE * sizeof(DTYPE), NULL, &status);
    checkError(status, "Failed to create buffer for data in layer");

    // Output results buffers
    output_buf[i*input_config[batch_size]+j] = clCreateBuffer(context, CL_MEM_READ_WRITE, OUT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
    checkError(status, "Failed to create buffer for output");
~~~

Here we create the data and output buffers, for each layer and for each input of the batch. 

~~~
    fc_1_buf[i] = clCreateBuffer(context,  CL_MEM_READ_WRITE, FC_BUF_SIZE * sizeof(DTYPE), NULL, &status);
    checkError(status, "Failed to create buffer for data in fc layer");

    fc_2_buf[i] = clCreateBuffer(context,  CL_MEM_READ_WRITE, FC_BUF_SIZE * sizeof(DTYPE), NULL, &status);
    checkError(status, "Failed to create buffer for data in fc layer");
~~~

As it can be seen, we create two fully connected buffers, `fc_1_buf` and the `fc_2_buf`. We do not create these buffers for each layers. If multiple layers are using these buffers, then we will alternate over these buffers. 

~~~
    if(j<CONV_NUM)
        iter_num = input_config[batch_size]; // for conv layers, process by batch_size time
    else
        iter_num = 1; // for FC layers, process only one time
~~~

The iteration number value is just the size of the batch. In our experiments, we mostly work with one item, which we always consider it as `1`.

Further we will iterate over all the items and move forward.

~~~
    conv_group_num_dim1   = ceil((float)layer_config[j][conv_x]/CONV_GP_SIZE_X);
    conv_group_num_dim2   = ceil((float)layer_config[j][conv_y]/CONV_GP_SIZE_Y);
    if(layer_config[j][conv_x]==1){
        conv_win_size_dim1  = layer_config[j][weight_w];
        conv_group_rem_dim1   = layer_config[j][weight_w];
    }
    else{
        conv_win_size_dim1  = layer_config[j][weight_w]+(CONV_GP_SIZE_X-1)*layer_config[j][conv_stride];
        if(layer_config[j][conv_x]%CONV_GP_SIZE_X==0)
            conv_group_rem_dim1   = CONV_GP_SIZE_X*layer_config[j][weight_w];
        else
            conv_group_rem_dim1   = layer_config[j][conv_x]%CONV_GP_SIZE_X*layer_config[j][weight_w];
    }
    conv_win_size_dim2    = layer_config[j][weight_h];
    conv_group_rem_dim2   = layer_config[j][weight_h];
    conv_win_size_dim1x2x3  = conv_win_size_dim1*conv_win_size_dim2*layer_config[j][weight_n];
    conv_group_rem_dim1x2x3 = conv_group_rem_dim1*conv_group_rem_dim2*layer_config[j][weight_n];

    weight_dim4_div_LaneNum = layer_config[j][weight_m]/LANE_NUM;
    data_dim1x2 = layer_config[j][data_w]*layer_config[j][data_h];
    weight_dim1x2 = layer_config[j][weight_w]*layer_config[j][weight_h];
    weight_dim1x2x3 = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n];

    if (j != 0) {
        pool_dim3_lastlayer_div_VecSize = layer_config[j-1][pool_z] / VEC_SIZE;
        conv_dim3_lastlayer_div_VecSize = layer_config[j-1][conv_z] / VEC_SIZE;
    }
~~~

For the `conv_group` and the `conv_dim` values, I do not really understand what is the purpose of them. We will get to them while analyzing the kernels. the last two parameters, `pool_dim3_lastlayer_div_VecSize` and `conv_dim3_lastlayer_div_VecSize` are just the helper values, while deploying the serializer and the deserializer kernels. 

Later, we first set the arguments for the deserialization kernel. This kernel is being launched for all the layers, except the first layer. 

~~~
    // Select the kernel input mem object source
    // data_buf -> conv1 -> output_buf -> lrn1 -> data_buf -> conv2 -> output_buf -> lrn2 -> data_buf
    // -> conv3 -> output_buf -> conv4 -> output_buf -> ...
    if(layer_config[j][memrd_src]==0) {
        status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
        checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
    } else if(layer_config[j][memrd_src]==1) {
        status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
        checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
    } else if(layer_config[j][memrd_src]==2) {
        status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &fc_1_buf[i]);
        checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
    } else {
        status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &fc_2_buf[i]);
        checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
    }
~~~

This is to set which input the `memRead` should read from. We are alternating between the `data_buf` and the `output_ buf`, and also the `fc_1_buf` and the `fc_2_buf`. The mechanism to choose the right buffer is not being done automatically, rather being specified by the user in the configuration parameter header. 

~~~
    conv_loop_cnt = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]/VEC_SIZE;
    conv_output_num = layer_config[j][conv_x]*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM; // new weight_m is divisible by LANE_NUM
    conv_control = (layer_config[j][conv_relu]&0x01)|(((~layer_config[j][pool_on])&0x01)<<1);
~~~

The `conv_loop_cnt` should simply be the total number of iterations that the convolutions will go through. it is just the `width x height x channels`, divided by the `VEC_SIZE`. The `conv_output_num` is the total number of outputs of the convolution. 

~~~
    pool_input_num = layer_config[j][conv_x]*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM; // new weight_m is divisible by LANE_NUM
    pool_line_size = layer_config[j][conv_x];
~~~

These specify the number of inputs for the pooling layer, and the size of the pool line.

~~~
    unsigned char batch_size_in_dim_log;
    unsigned char mask = 0xff;
    unsigned char memWr_dim1, memWr_dim2;
    unsigned short memWr_dim3;
~~~

The `batch_size_in_dim_log` is still a non-recognizable thing for me. The `mask`, I do not fully understand. The `memWr_dim1`, `memW_dim2`, and the `memWr_dim3` are set later. The value of these dimensions depend on the type of the last layer, which is either a convolution or a pooling. 

~~~
	if(layer_config[j][pool_on]==1){
	    memWr_dim1 = layer_config[j][pool_x];
		memWr_dim2 = layer_config[j][pool_y];
		memWr_dim3 = layer_config[j][pool_z];
	}
	else{
		memWr_dim1 = layer_config[j][conv_x];
		memWr_dim2 = layer_config[j][conv_y];
		memWr_dim3 = layer_config[j][conv_z];
	}
~~~

Here the three operating dimensions of the `memWr` is being specified, depending on the existence of the pooling layer.

For a few lines, the codes sets the arguments for the `memWr` kernel and the other kernels.

~~~
	status = clEnqueueTask(que_memRd[i], knl_memRd[i], 0, NULL, &memRd_event[i]);
    checkError(status, "Failed to launch kernel memRD kernel");
~~~

Pushing the `memRd` kernel into the queue. 

~~~
	status = clEnqueueTask(que_conv[i], knl_conv[i], 0, NULL, &conv_event[i]);
    checkError(status, "Failed to launch kernel conv kernel");
~~~

Pushing the `convolution` kernel into the respective queue. 

~~~
	if(layer_config[j][pool_on]){
		status = clEnqueueTask(que_pool[i], knl_pool[i], 0, NULL, &pool_event[i]);
		checkError(status, "Failed to launch kernel pooling");
		if(k == 0&&pic_num==1)
			printf("\nLaunching single work-item kernel Pooling\n");
    }
~~~

Pushing the `pooling` kernel into the respetive queue, if it is activated.

~~~
	knl_memWr_global_size[0] = memWr_dim1;
	knl_memWr_global_size[1] = memWr_dim2;
	knl_memWr_global_size[2] = layer_config[j][weight_m]; // pool_z equals original weight_m, new weight_m is divisible by LANE_NUM
	knl_memWr_local_size[0] = 1;
	knl_memWr_local_size[1] = 1;
    knl_memWr_local_size[2] = LANE_NUM;
~~~

Setting the local and global work size information for the `memWr` kernel. 

~~~
	if(layer_config[j][lrn_on]){

		knl_lrn_global_size[0] = layer_config[j][pool_x];
		knl_lrn_global_size[1] = layer_config[j][pool_y];
		knl_lrn_global_size[2] = layer_config[j][pool_z]/VEC_SIZE;
		knl_lrn_local_size[0] = 1;
		knl_lrn_local_size[1] = 1;
		knl_lrn_local_size[2] = layer_config[j][pool_z]/VEC_SIZE;

		if(k == 0&&pic_num==1)
			printf("\nLaunching kernel lrn with local size: %d, %d, %d  (global size: %d, %d, %d)\n", (int)knl_lrn_local_size[0], (int)knl_lrn_local_size[1], (int)knl_lrn_local_size[2], (int)knl_lrn_global_size[0], (int)knl_lrn_global_size[1], (int)knl_lrn_global_size[2]);

		status = clEnqueueNDRangeKernel(que_memWr[i], knl_lrn[i], 3, NULL, knl_lrn_global_size, knl_lrn_local_size, 0, NULL, &lrn_event[i]);
		checkError(status, "Failed to launch kernel lrn");
    }
~~~

Setting the local and the global work size for the `lrn` kernel, if it is enabled. 

~~~
	if(layer_config[j][lrn_on]){
		status = clWaitForEvents(num_devices, lrn_event);
		checkError(status, "Failed to finish lrn event");
	}
	else{
		status = clWaitForEvents(num_devices, memWr_event);
		checkError(status, "Failed to finish memWR event");
    }
~~~

waiting for the especific events for the current layer execution, before moving any further. 

~~~
	memRd_time[j] += getKernelStartEndTime(memRd_event[i], "memRd");
	conv_time[j]  += getKernelStartEndTime(conv_event[i], "conv");
	if(layer_config[j][pool_on])
		pool_time[j] += getKernelStartEndTime(pool_event[i], "pool");
	memWr_time[j] += getKernelStartEndTime(memWr_event[i], "memWr");
	if(layer_config[j][lrn_on])
        lrn_time[j] += getKernelStartEndTime(lrn_event[i], "lrn");
~~~

Calculating the processing time of each single layer. 

~~~
	readDataBack();
    verifyResult(pic_num);
~~~

Reading back the data from the respetive last later and verify the numbers. 


## Read Back Data

~~~
	if(LAYER_NUM<CONV_NUM){ // verify conv results
		read_buf_size = output_config[output_w]*output_config[output_h]*output_config[output_n];
	}
	else // verify the last conv and all fc results
        read_buf_size = output_config[output_w]*output_config[output_h]*output_config[output_n]*input_config[batch_size];
~~~

calculating the buffer size of the data that we read back. If the last layer is a convolution, then it is only one image. Otherwise, it is the size of one image, multiplied by the `batch_size`. 

~~~
	if(layer_config[LAYER_NUM-1][memwr_dst] == 2){
		printf("\nCopyed all batched results from fc_1 buffers.\n");
		status = clEnqueueReadBuffer(que_memWr[0], fc_1_buf[0], CL_FALSE,          // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}
	else if(layer_config[LAYER_NUM-1][memwr_dst] == 3){
		printf("\nCopyed all batched results from fc_2 buffers.\n");
		status = clEnqueueReadBuffer(que_memWr[0], fc_2_buf[0], CL_FALSE,          // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}
	// For other layers, read results from data and output buffers
	else if(layer_config[LAYER_NUM-1][memwr_dst]^layer_config[LAYER_NUM-1][lrn_on]){// if lrn is used, the mem dst is changed back to src
		printf("\nCopyed one result from NO.%d output buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], output_buf[batch_item_num], CL_FALSE,         // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}
	else{
		printf("\nCopyed one results from NO.%d data buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], data_buf[batch_item_num], CL_FALSE,           // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
    }
~~~

figuring out the buffer that holds the result. it is either `data_buf`, or `output_buf`, or `fc_1_buf`, or `fc_2_buf`. 

## Verify Result

Checking the reordered output against the golden reference and figuring out whether it has been matched or not.

## Load Image to Buffer

~~~
    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);

    	bin_file_r.read((char *)image, sizeof(DTYPE)*IMAGE_FILE_SIZE);
    	printf("\n%d bytes image data read from binary files\n", file_size);
		if(IMAGE_FILE_SIZE!=(file_size/(sizeof(DTYPE))))
			printf("Warning: image file size does not match user configuration !!!\n");
    	bin_file_r.close();
    }
    else
		printf("Image file does not exits !!!\n");
~~~

Opening the image file and load it into the buffer as a byte array. 
    
~~~
    for(unsigned n = 0; n<layer_config[0][data_n]/VEC_SIZE; n++){
		for(unsigned i = 0; i<layer_config[0][data_h]; i++){
			for(unsigned j = 0; j<layer_config[0][data_w]; j++){
				for(unsigned k = 0; k<VEC_SIZE; k++){
					if((n*VEC_SIZE+k)<layer_config_original[0][data_n]){ //  when layer_config[0][data_n] > layer_config_original[0][data_n], only copy valid pixels
						data_init[n*VEC_SIZE*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w]*VEC_SIZE + j*VEC_SIZE + k]
							= (DTYPE) image[(n*VEC_SIZE+k)*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w] + j];
					}
				}
			}
		}
    }
~~~

Reordering the input data into the `data_init`, so it could be fed properly to the model. In this model, the data is being vectorized on the `input channel`. We should remember that the same vectorization has been done also on the weights and biases values. 

~~~
	status = clEnqueueWriteBuffer(que_memRd[i], data_buf[i*input_config[batch_size]+j], CL_TRUE, 0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
    checkError(status, "Failed to transfer input image");
~~~

Writing the input data into the `data_buf`, so it can be used by the model for the processing.



