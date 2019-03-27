/////////////////////////////////////////////
//Name: Template accelerator module
//Description: The purpose of this module
//is to enable an algo to recieve and send
//N streams of input and M streams of output.
/////////////////////////////////////////////
#include "cluster_platform.h"

/////////////////////////////////////////////
/////////////////ALGO SECTION////////////////
/////////////////////////////////////////////
__attribute__((num_compute_units(ALGO_COPY)))
__kernel void algo(
                   __global ulong * restrict algo_config,
                   __global ulong * restrict i_parm,
                   __global ulong * restrict o_parm
                  )
{
  //create rdma packet
  tx_type_req TxRdmaReq;
  rx_type_resp ingress_ctrl;
  //get compute ID
  int c_id = get_compute_id(0);
 
  //determines when the algo is ready to run!!!
  ingress_ctrl = read_channel_intel(rx_rdma_res);

  //update output 1
  for(int i=0; i < ingress_ctrl.data_len[0]; i++)
    o_parm[i] = i_parm[i] + 1;

  //SEND LOGIC CONTROL THE FOLLWOING:
  //TxRdmaReq.valid = (true, false)
  //TxRdmaReq.(parm_id,start,end) = (32bit int)
  
  //send request to rdma arbiter
  TxRdmaReq.valid = true;
  TxRdmaReq.start = 0;
  
  TxRdmaReq.parm_id = 0;
  TxRdmaReq.end = ingress_ctrl.data_len[0]-1;
  write_channel_intel (tx_rdma_req[c_id], TxRdmaReq);

}

