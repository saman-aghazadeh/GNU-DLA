/////////////////////////////////////////////
//Name: Template accelerator module
//Description: The purpose of this module
//is to enable an algo to recieve and send
//N streams of input and M streams of output.
/////////////////////////////////////////////
#include "template.h"

/////////////////////////////////////////////
///////////////GENERAL SECTION///////////////
/////////////////////////////////////////////

//number of outputs states
void _move_data_to_io(tx_type_req req, unsigned long dst, unsigned long src, unsigned long dst_input,  __global unsigned long *parm)
{
  tx_type_io TxIoData;
  unsigned long index = 0;
  int c_id = get_compute_id(0);
  unsigned long end = (HDR_SIZE + req.end);
  unsigned long before_end = end - 1;

  for(index = req.start; index < end; index++)
  {
    switch(index)
    {
      case TX_HDR_DST:    TxIoData.data = dst; break;
      case TX_HDR_SRC:    TxIoData.data = src; break;
      case TX_HDR_CTRL:   TxIoData.data = DATA_CTRL; break;
      case TX_HDR_INPUT:  TxIoData.data = dst_input; break;
      case TX_HDR_OFFSET: TxIoData.data = req.start; break;
      default:            TxIoData.data = parm[index - HDR_SIZE]; break;
    }

    //data passing through
    TxIoData.last_data = (index == before_end)?1:0;

    write_channel_intel (tx_data_stream[c_id], TxIoData);
  }

}


__attribute__((num_compute_units(ALGO_COPY)))
__kernel void tx_rdma_arb(
                          __global unsigned long * restrict sys_config,
                          __global ulong * restrict o_parm 
                         )
{
  tx_type_req TxRdmaReq;

  while (1)
  {
    //dump all packets that don't have valid data 
    TxRdmaReq.valid = false;
    while (!TxRdmaReq.valid)
      TxRdmaReq = read_channel_intel(tx_rdma_req[0]); 

    //case statement to switch on inputs
    //last element special case:

    //Get the destination MAC address
    unsigned long dst = sys_config[NUM_OUTPUT_PARMS];

    //Get the source MAC address
    unsigned long src = sys_config[NUM_OUTPUT_PARMS*2];

    //Get input ID for the down stream
    unsigned long dst_input = sys_config[NUM_OUTPUT_PARMS*3];

    _move_data_to_io(TxRdmaReq, dst, src, dst_input, o_parm);
  }
}

