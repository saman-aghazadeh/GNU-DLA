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

//Current FPGA MAC ADDRESS//
#define DST_MAC_ADDRESS 0x0000102C00D70C00

////////////////////////////////////////////
//
//Filters out all packets that are not 
//
////////////////////////////////////////////
__kernel void rx_io_channel()
{

  ulong2 MAC_Data;
  rx_type_io new_data;
  unsigned long drop[3];
  bool end_of_packet=false; 
  bool start= false;
  unsigned long cnt =0; //represents the count after the packet is valid
  drop[0]=1; drop[1]=1; drop[2]=1;

  while(1)
  {
     //read ethernet pipeline
     MAC_Data = read_channel_intel(rx_eth_ch0);

     start = (((MAC_Data.y) & (SOP|EOP)) == SOP) &&
             ((MAC_Data.x & 0x0000FFFFFFFFFFFF) == DST_MAC_ADDRESS);

     //end of packet stream
     end_of_packet = ((((MAC_Data.y) & (SOP|EOP)) == EOP));


     drop[2] = start?0:1;

     //latch valid flag through the stream
     if( (drop[0] == 0) && ((MAC_Data.y) & (SOP|EOP)) != NULL_PACKET)
     {
       new_data.valid = 1;
     }
     else
     {
       new_data.valid = 0;
     } 

     //indicates to the RDMA engine the last packet in the stream
     new_data.last = (end_of_packet)?1:0; 

     //stage the new data to the rdma engine
     new_data.data = MAC_Data.x;

     //latch packet type between for entire stream
     //can keep adding definitions to latch variables by adding cnt

     new_data.pkt_type = (cnt < 1)?( MAC_Data.x  & 0x000000000000FFFF):new_data.pkt_type; 

     if( new_data.valid == 1 )
     {
       if(new_data.pkt_type == DATA_CTRL)
       {
         write_channel_intel(rx_data_stream, new_data);
       }
       else if( (new_data.pkt_type == ALGO_CTRL) ||
                (new_data.pkt_type == SYS_CTRL) )
       {
       
         write_channel_intel(rx_ctl_stream, new_data);
       }

     }

     //keep shifting until drop isn't low 
     //keep shifting until SRC and DST are out the way
     if( drop[0] == 1 ) 
     {
       drop[0] = drop[1];
       drop[1] = drop[2];
     }

     //make shift counter state machine
     cnt = ( (new_data.valid == 1) )?(cnt + 1):cnt; 


     if(end_of_packet)
     {
       drop[0] = 1;
       cnt     = 0;
     }

  }  

}
