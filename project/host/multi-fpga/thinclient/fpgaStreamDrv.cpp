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
#include "fpgaStreamDrv.h"

#define FPGA_MTU_SIZE 1024 

using namespace std;
namespace fpga{

union S{
 char byte_data[8];
 short int address[4];
 int value[2];
 unsigned long final_num;
}; 


void util_flip_bytes(S& data)
{
  //reverse order
  S temp = data;
  for(int i=0; i < sizeof(data); i++)
    data.byte_data[i] = temp.byte_data[7-i];
}

int setup_comm(string src_mac, string dst_mac, string tag, fpga_meta * handle)
{
    if(handle == nullptr)
    {  
      cout <<"Error: Invalid fpga_meta handle"<<endl;
      return -1;
    }
    handle->src_mac = src_mac;
    handle->dst_mac = dst_mac;
    handle->type = tag;
    
    /*prepare sockaddr_ll*/
    /*RAW communication*/
    handle->socket_address.sll_family = PF_PACKET;	
    /*we don't use a protocoll above ethernet layer
     *   ->just use anything here*/
    handle->socket_address.sll_protocol = htons(ETH_P_IP);	

    /*index of the network device
     * see full code later how to retrieve it*/
    //em1  =  2
    //em2  =  3
    //p1p2 =  4
    //p2p2 =  5
    handle->socket_address.sll_ifindex  = 3;

    /*ARP hardware identifier is ethernet*/
    handle->socket_address.sll_hatype   = ARPHRD_ETHER;
	
    /*target is another host*/
    handle->socket_address.sll_pkttype  = PACKET_OTHERHOST;

    /*address length*/
    handle->socket_address.sll_halen    = ETH_ALEN;		
    /*MAC - begin*/
    handle->socket_address.sll_addr[0]  = dst_mac[0];		
    handle->socket_address.sll_addr[1]  = dst_mac[1];		
    handle->socket_address.sll_addr[2]  = dst_mac[2];		
    handle->socket_address.sll_addr[3]  = dst_mac[3];		
    handle->socket_address.sll_addr[4]  = dst_mac[4];		
    handle->socket_address.sll_addr[5]  = dst_mac[5];		
    /*MAC - end*/
    handle->socket_address.sll_addr[6]  = 0xAA;/*not used*/
    handle->socket_address.sll_addr[7]  = 0xAA;/*not used*/

    //Step 1: open a socket SOCK_STREAM, SOCK_DGRAM, SOCK_RAW
    handle->socket = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (handle->socket < 0) {
      cout <<"Error opening socket"<<endl;
      return -1;
    }

    return 0;
}


int write_data(fpga_meta handle, string tag, const void * data_in, long size)
{
    long send_results=0;
    long remaining = size;
    //create packets of 1024b
    long packet_num = size/1024;
    //concat data into 
    string fcs({0x00,0x00,0x00,0x00});
    //need to create packets
    //create header information
    string header  =  handle.dst_mac + handle.src_mac;
           header  += tag.empty()?handle.type:tag;
           header  += {0x00, 0x00};

    string data_seg="";
    for(int i=0; i <= packet_num; i++){
      cout <<"Sending Packet#"<<i<<" out of "<<packet_num<<std::endl;
      for(int j=0; j < std::min(remaining,(long)1024); j++){
        long index = i*1024+j;
	data_seg += ((char*) data_in)[index];
      }

      //construct packet
      string send_data = header + data_seg + fcs;
      cout<<"Sending data (size="<<send_data.length()<<")..."<<std::endl;
      send_results += sendto(handle.socket,(const void *) send_data.c_str(), send_data.length(), 0, 
  	        (struct sockaddr*)&(handle.socket_address), sizeof(handle.socket_address));
      //keeps track of remaining packets
      remaining -= send_results;    
      if( send_results != send_data.length()){
        cout <<"Packet#: "<<i<<"Error sending out data(wrote: "<<send_results<<" out of "<<size<<std::endl;
        return -1; 
      }
    } //end of packet for loop
   //SUCCESS
   return 0;
}

int write_stream_data(fpga_meta handle, string tag, const void * data_in, long size)
{
    unsigned long send_results=0;
    unsigned long remaining = 0;
    unsigned long protocol_hdr_sz = 2*sizeof(unsigned long);
    //create packets of 1024b
    unsigned long packet_num = std::ceil((size-protocol_hdr_sz)/1024.0);
    //concat data into 
    string fcs({0x00,0x00,0x00,0x00});
    //skipping first two 
    const char * data = (const char *) data_in;

    //converting into string
    string s_data(data, size);

    auto pad = [](unsigned long data_in)->string{
       string s1 ="";
       unsigned long data = data_in;
       for(int i=0; i < sizeof(unsigned long); i++)
         s1 += ((char*) &data)[i]; 
       return s1;
    };

    //create header information
    string header  =  handle.dst_mac + handle.src_mac;
           header  += tag.empty()?handle.type:tag;
           header  += {0x00, 0x00};

    string protocol_hdr     = s_data.substr(0, protocol_hdr_sz);  
    string data_segment     = s_data.substr(protocol_hdr_sz, s_data.length());  
    remaining = (size - protocol_hdr_sz) + (header.length() + protocol_hdr_sz)*packet_num;
    unsigned long remaining_data_size = data_segment.length();
    cout << "data size = "<< remaining_data_size<<" size= "<<size<<std::endl;
    for(int i=0; i < packet_num; i++){
      cout <<"Sending Packet#"<<i<<" out of "<<packet_num<<std::endl;
      //construct packet
      unsigned long offset = (i*FPGA_MTU_SIZE/sizeof(unsigned long));
      string s_offset((char *) &offset, sizeof(unsigned long));

      /*unsigned long max_data_block_sz = FPGA_MTU_SIZE - 
                                        (header.length() + 
                                         protocol_hdr.length() +
                                         s_offset.length() +
                                         fcs.length());*/

      unsigned long max_data_block_sz = FPGA_MTU_SIZE; 
      unsigned long min_data_sz = std::min(remaining_data_size, max_data_block_sz);
        
      cout <<"min_data_sz =" << min_data_sz << std::endl; 
      string send_data = header + 
                         protocol_hdr + s_offset +
                         data_segment.substr(i*(max_data_block_sz), min_data_sz) + 
                         fcs;
 
      cout<<"Sending data (size="<<send_data.length()<<")..."<<std::endl;
      send_results = sendto(handle.socket,(const void *) send_data.c_str(), send_data.length(), 0, 
  	        (struct sockaddr*)&(handle.socket_address), sizeof(handle.socket_address));

      //keeps track of remaining packets
      remaining -= send_results;   
      remaining_data_size -= min_data_sz;
 
      if( send_results != send_data.length()){
        cout <<"Packet#: "<<i<<"Error sending out data(wrote: "<<send_results<<" out of "<<size<<std::endl;
        return -1; 
      }

      //sleep(1);
    } //end of packet for loop

    if( remaining_data_size != 0)
      printf("Still have data to send\n");
   //SUCCESS
   return 0;
}

int write_stream_data2(fpga_meta handle, string tag, const void * data_in, long size)
{
    unsigned long send_results=0;
    unsigned long remaining = 0;
    //create packets of 1024b
    unsigned long packet_num = std::ceil(size/1024.0);
    char * data = (char *) data_in;
    int protocol_offset = 0;

    //concat data into 
    string fcs({0x00,0x00,0x00,0x00});
    //need to create packets
    //create header information
    string header  =  handle.dst_mac + handle.src_mac;
           header  += tag.empty()?handle.type:tag;
           header  += {0x00, 0x00};
    
    auto concat = [&] (int index)->string{
      char * num = &data[index*sizeof(unsigned long)];
      string s1="";
      for(int i=0; i < sizeof(unsigned long); i++)
        s1 += num[i];
      return s1;
    };

    auto pad = [](unsigned long data_in)->string{
       string s1 ="";
       unsigned long data = data_in;
       for(int i=0; i < sizeof(unsigned long); i++)
         s1 += ((char*) &data)[i]; 
       return s1;
    };

    auto byte = [] (unsigned long in)->size_t{ return in*sizeof(unsigned long); };

    protocol_offset = 3;
    string hdr_pkt_type = concat(0);
    string hdr_chan_id  = concat(1);
    string protocol_hdr = hdr_pkt_type + hdr_chan_id;
    data = &data[byte(protocol_offset-1)];
    remaining = (size - byte(protocol_offset-1)) + (header.length() + byte(protocol_offset))*packet_num;
    size = remaining;


    string data_seg="";
    string send_data = "";
    for(int i=0; i < packet_num; i++){
      cout <<"Sending Packet #"<<i<<" out of "<<packet_num<<std::endl;
      data_seg.clear();
      data_seg = protocol_hdr + pad(i*FPGA_MTU_SIZE);
      //data_seg = protocol_hdr;
      unsigned long end =std::min((remaining - byte(protocol_offset) - header.length() - fcs.length()),
                                 (unsigned long) 
                                 (FPGA_MTU_SIZE - byte(protocol_offset) - header.length() - fcs.length()));

      for(int j=0; j < end; j++){
        unsigned long index = i*FPGA_MTU_SIZE+j;
	data_seg += data[index+byte(protocol_offset)];
      }

      //construct packet
      send_data.clear();
      send_data = header + data_seg + fcs;
      send_results = sendto(handle.socket,(const void *) send_data.c_str(), send_data.length(), 0, 
  	        (struct sockaddr*)&(handle.socket_address), sizeof(handle.socket_address));
      //keeps track of remaining packets
      remaining -= send_results;    
      cout<<"Sent data (size="<<send_data.length()<<", remaining="<<remaining <<")..."<<std::endl;
    } //end of packet for loop

    if( remaining != 0 ){
      cout <<" Error sending out data(wrote: "<<size - remaining<<" out of "<<size<<std::endl;
      cout <<"Remaining Btyes: "<<remaining<<std::endl;
      return -1; 
    }
   //SUCCESS
   return 0;
}
int read_data(fpga_meta handle, void * data_out, long size_in)
{
  return recv(handle.socket, data_out, size_in, 0);
}

int write_ctl(fpga_meta handle, string tag, short int address, int value, bool reset)
{
  S data = {.final_num =0};
  
  data.value[0]     = value;
  data.address[2]   = address | 0xC000;
  data.byte_data[6] = reset?0x40:0x07; 
  printf("Sending: 0x%lx\n",data.final_num);
  
  //reverse order
  S temp = {.final_num =0};
  for(int i=0; i < sizeof(data); i++)
    temp.byte_data[i] = data.byte_data[7-i];

  return write_data(handle, tag, (const void*) temp.byte_data, sizeof(temp));

}


int write_region(fpga_meta handle, region_types region, unsigned long addr, unsigned long value, bool reverse)
{

  S c_region = {.final_num =region};
  S c_addr = {.final_num =addr};
  S c_val = {.final_num =value};

  if( reverse ) 
  {
    util_flip_bytes(c_region);
    util_flip_bytes(c_addr);
    util_flip_bytes(c_val);
  }

  printf("Setting: s_region = 0x%lx,  s_addr = 0x%lx,  s_val = 0x%lx\n",
         c_region.final_num, c_addr.final_num, c_val.final_num);

  string s_region(c_region.byte_data, 8);
  string s_addr  (c_addr.byte_data,   8);
  string s_val   (c_val.byte_data,    8);

  string final_cmd= s_region + s_addr + s_val;
  return write_data(handle, "", final_cmd.c_str(), 3*sizeof(unsigned long));
}



//close socket
void close_stream(fpga_meta handle){close(handle.socket);}


} //end of namespace



