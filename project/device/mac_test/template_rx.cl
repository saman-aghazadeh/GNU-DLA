/////////////////////////////////////////////
//Name: Template accelerator module
//Description: The purpose of this module
//is to enable an algo to recieve and send
//N streams of input and M streams of output.
/////////////////////////////////////////////
#include "system_reg.h"
#include "template.h"

/////////////////////////////////////////////
///////////////GENERAL SECTION///////////////
/////////////////////////////////////////////

__kernel void rx_data_arb(
                          __global ulong * restrict sys_config,
                          __global ulong * restrict i_parm 
                         )
{
  //input data from io kernel
  rx_type_io ingress; 
  rx_type_resp outgress_ctrl;
  unsigned int state=0;
  unsigned long addr=0; 

  //configuration counter
  //data index for each parameter
  unsigned long data_index=0;
  unsigned long data_lim;
  unsigned long data_len;
 
  data_len = 0;
  data_lim = 1;

  while(1)
  {  
    //read data from io kernel
    //data should be clean directed to this FPGA
    ingress = read_channel_intel(rx_data_stream);
   
    //base address
    data_index = (state < 3)?ingress.data:data_index;
 
    //load new limits for each variable 
    data_lim = (state == 1)?sys_config[0]:data_lim;

    //advance state - 4th element is the first valid piece of information   
    ++state;
   
    //barrier 
    if( state < 4 ) continue;

    //////////////////////////////////////////////////////////////////////////////////////
    i_parm[data_index] = ingress.data;

    //increment the length for each parameter
    data_index++;
    data_len = data_index;

    //////////////////////////////////////////////////////////////////////////////////////
    //check to see if all the inputs hit there limits
    // if so generate request to algo 
    if( data_len >= data_lim )
    {
      outgress_ctrl.data_len[0] = data_len;
      data_len = 0; //reset len for next transmission

      write_channel_intel(rx_rdma_res, outgress_ctrl);
 
    }
    state = (ingress.last == 1)?0:state;
    //////////////////////////////////////////////////////////////////////////////////////   
  } //end of while loop
}

__kernel void rx_ctrl_arb(
                          __global ulong * restrict sys_config,
                          __global ulong * restrict algo_config
                         )
{

  rx_type_io ingress;
  unsigned int state=0;
  unsigned long addr=0; 

  while(1)
  {
    //read data from io kernel
    //data should be clean directed to this FPGA
    ingress = read_channel_intel(rx_ctl_stream);

    if( state == 1)
    {
      addr = ingress.data;
      state++;
    }
    else if( state == 2)
    {
      switch(ingress.pkt_type)
      {
        case ALGO_CTRL: algo_config[addr] = ingress.data; break;
        case SYS_CTRL:  sys_config [addr] = ingress.data; break;
        default: break;
      }
      state++;
    }
    else
    { //reset state 
      state++;
      state = (ingress.last == 1)?0:state;
    } 
    
  }

}

