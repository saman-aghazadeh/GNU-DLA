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

__kernel void tx_io_channel()
{
  tx_type_io  TxIoData;
  unsigned long hdr[HDR_SIZE];
  ulong2 MAC_Data;
  unsigned char  sos = 1;
  unsigned long bytes=0;
  unsigned long ph1_bytes=0;
  unsigned long cnt=0;
  unsigned char eop[2];

  eop[0] = 0;
  eop[1] = 0;

  while(1) 
  {
    //tracks end of packet
    if( (bytes-ph1_bytes) >= MTU_SIZE){
      eop[0] = 1;
      ph1_bytes = bytes; 
      hdr[TX_HDR_OFFSET] = bytes;
    }
    else{
      eop[0] = 0;
    }
 
    //reads channel from move_data_to_io  
    TxIoData = read_channel_intel (tx_data_stream[0]);
   
    if( cnt < HDR_SIZE)
    {
       //index = 0, DST
       //index = 1, SRC + TYPE
       //index = 2, CTRL or DATA
       //index = 3, INPUT ID
       //index = 4, OFFSET
       hdr[cnt] = TxIoData.data;
    }

    //controls whether the packet size limit is reached
    //eop[1] == 0 indicates that the packet hasn't reached the max

    for(int i=0; i <= HDR_SIZE; i++)
    {
      if(eop[1] == 0)
      {
        //fill control + data to MAC
        MAC_Data.y = sos | (TxIoData.last_data << 1) | (eop[0] << 1);
        MAC_Data.x = TxIoData.data;
      }
      else
      {
        MAC_Data.y = (i==0)?1:0;
        MAC_Data.x = (i<HDR_SIZE)?hdr[i]:TxIoData.data;
      }

      write_channel_intel( tx_eth_ch0, MAC_Data);
      
      //break out of the for loop
      if(eop[1] == 0) break;
     
    }

    //prepare for the next transaction 
    sos = TxIoData.last_data;
 
    //increment the index   
    if( sos == 1) {
      cnt = 0;
      bytes = 0;  //reset index
      ph1_bytes = 0;
    }
    else {
      cnt++;
      bytes = bytes+8;
    }

    eop[1] = eop[0];
  }
}

