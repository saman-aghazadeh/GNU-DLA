#ifndef TEMPLATE_H
#define TEMPLATE_H

#pragma OPENCL EXTENSION cl_intel_channels : enable

////////////////////////////////////////////
//Type defes
enum TX_HEADERS {TX_HDR_DST=0, TX_HDR_SRC, TX_HDR_CTRL, TX_HDR_INPUT, TX_HDR_OFFSET};
enum ctrl_states {ALGO_CTRL= 0, SYS_CTRL, LOAD_CTRL, DATA_CTRL};
/////////////////////////////////////////////
/////////////////RX SECTION//////////////////
/////////////////////////////////////////////

typedef struct rx_type_resp {
                             ulong data_len[NUM_INPUT_PARMS]; 
                            } rx_type_resp;

typedef struct rx_type_io {
			   ulong pkt_type;
			   ulong valid;    
			   ulong last;    
			   ulong data;
			  } rx_type_io;

typedef struct rx_ctrl_type {
                             bool sw;    
                             ulong addr;
                            } rx_ctrl_type;
/////////////////////////////////////////////
////////////////RX Channel///////////////////
//Desc: PHY-to-kernel 
//control pipe (between phy-io) 
channel ulong2 rx_eth_ch0 
               __attribute__((depth(1024))) 
               __attribute__((io("input_ch0")));

//Desc: Kernel-to-kernel 
//control pipe (between io-filter) 
channel rx_type_io rx_data_stream
                   __attribute__((depth(16)));
channel rx_type_io rx_ctl_stream
                   __attribute__((depth(16)));

//Desc: Kernel-to-kernel 
//control pipe (between filter-Algo) 
channel rx_type_resp rx_rdma_res 
                     __attribute__((depth(1)));

/////////////////////////////////////////////
/////////////////TX SECTION//////////////////
/////////////////////////////////////////////
typedef struct tx_type_req {
                           bool valid;
                           int parm_id;
                           int start;    
                           int end; 
                         } tx_type_req;
typedef struct tx_type_io {
			   unsigned char last_data; 
			   ulong data;    
			 } tx_type_io;

/////////////////////////////////////////////
////////////////TX Channel///////////////////
//Desc: Kernel-to-kernel 
//control pipe (between algo-arb) 
channel tx_type_req tx_rdma_req [ALGO_COPY]
                    __attribute__((depth(1)));
//////////////////////////
//Desc: Kernel-to-kernel control pipe (between arb-io)
//      first ulong is destination address
channel tx_type_io tx_data_stream[TX_CHANNELS] 
                   __attribute__((depth(1)));

//////////////////////////
//Desc: Kernel-to-io control pipe (between io-phy)
channel ulong2 tx_eth_ch0  
               __attribute__((depth(4))) 
               __attribute__((io("output_ch0")));
#endif
