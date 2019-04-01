#include "fpgaStreamDrv.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <linux/if_ether.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <linux/if_packet.h>
#include <linux/if_arp.h>
#include <initializer_list>
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>


//int setup_comm(string src_mac, string dst_mac, string tag, fpga_meta * handle);
//int write_data(fpga_meta handle, string tag, void * data_in, long size);
//int read_data(fpga_meta handle, string tag, void * data_out, long size_in, long *size_out);
//void close_stream(fpga_meta handle);
using namespace fpga;
int main(int argc,const char *argv[])
{
     fpga_meta fpga1_meta, fpga2_meta;
     int nStatus =0;
     bool reverse = true;
     string src_mac({(char)0x14, (char)0x18, (char) 0x77, (char)0x33, (char)0xB0, (char) 0x6B});
     string fpga1_port8({(char)0x00, (char)0x0C, (char)0xD7, (char)0x00, (char)0x2D, (char)0x10});
     string fpga2_port5({(char)0x00, (char)0x0C, (char)0xD7, (char) 0x00, (char) 0x2D, (char) 0x11});
     string packet_type({(char)0xAA, (char)0xAA});
     char dummy_data[256];
     char rx_dummy_data[256];    
     unsigned long addr, value;
     //setup comm
     nStatus = setup_comm(src_mac, fpga1_port8, packet_type, &fpga1_meta);
     nStatus = setup_comm(src_mac, fpga1_port8, packet_type, &fpga2_meta);
     
    //fill dummy data
    for(int i=0; i < 256; i++)
      dummy_data[i] = i;
   
    //send data
    //nStatus = write_data(fpga1_meta, "", (const void *) dummy_data, 256);

    //read data
    //nStatus = read_data(fpga2_meta, rx_dummy_data, 256);

    addr =  0x0000000000000001;
    value = 0x0000000000000002;
    nStatus = write_region(fpga1_meta, ALGO_CTRL, addr, value, reverse); 

    addr =  0x0000000000000003;
    value = 0x0000000000000004;
    nStatus = write_region(fpga1_meta, SYS_CTRL, addr, value, reverse); 

    addr =  0x0000000000000005;
    value = 0x0000000000000006;
    nStatus = write_region(fpga1_meta, LOAD_CTRL, addr, value, reverse); 

    addr =  0x0000000000000007;
    value = 0x0000000000000008;
    nStatus = write_region(fpga1_meta, DATA_CTRL, addr, value, reverse); 

    //close sockets
    close_stream(fpga1_meta);
    close_stream(fpga2_meta);
   
    return 0;  
}
