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
#include <map>
#include <regex>

std::map<string, string> input_map;

union S{
 char byte_data[8];
 short int address[4];
 int value[2];
 long long final_num;
}; 


int parse_input(int argc, const char* argv[])
{
  for(int i=1; i < argc; i++)
  {
    std::string s1(argv[i]);
    regex equals("[^=]+", regex_constants::ECMAScript | regex_constants::icase);
    auto word_begin = sregex_iterator(s1.begin(), s1.end(), equals);
    auto word_end = sregex_iterator();

    if(std::distance(word_begin, word_end) != 2)
    {
      std::cout <<"rejected paraemter: "<<i<<"("<<argv[i]<<")"<<std::endl;
      return -1;
    }
    else
    {
      std::string key = word_begin->str();
      word_begin++;
      std::string value = word_begin->str();
      input_map.insert(std::pair<string, string>(key, value));
    }
  }

  return 0;
}

using namespace fpga;
int main(int argc,const char *argv[])
{
     fpga_meta fpga1_meta;
     int nStatus =0;
     std::string src_mac({(char)0x14, (char)0x18, (char) 0x77, (char)0x33, (char)0xB0, (char) 0x6B});
     std::string fpga1_port10({(char)0x00, (char)0x0C, (char)0xD7, (char)0x00, (char)0x2C, (char)0x10});
     std::string packet_type({(char)0xCC, (char)0xCB});
     char kernel_cntrl[8];
     const long cmd_len = 8;

     auto stohex = [&](std::string input)->int{
        return  (int) strtol(input.c_str(), NULL, 16);
     };

     if(parse_input(argc, argv) != 0)
     {
       std::cout <<"Invalid Inputs"<<std::endl;
       return -1;
     }
                 //14b    //1b   //32b  /4bits
     short int address; 
     int value;
     int reset=0, debug;
     S data = {.final_num = 0};

     //everything needs to be 8Byes then shifted
     address = stohex(input_map["address"]) | 0xC000;
     value   = stohex(input_map["value"]);
     reset   = stol(input_map["reset"]);
     debug   = stol(input_map["debug"]);
    
     //first 32-bits
     data.value[0] = value;
     //next 16-bits
     data.address[2] = address;
     //control 8-bits
     data.byte_data[7] = reset?0x08:0xE0;

     
     if(debug)
       for(int i =0; i < sizeof(S); i++)
         printf("byte %i: 0x%01x\n",i, data.byte_data[i] & 0x00FF);

     printf("Sending: 0x%lx\n",data.final_num);

     if(debug)
	return 0;

     //setup comm
     nStatus = setup_comm(src_mac, fpga1_port10, packet_type, &fpga1_meta);
     
     //send data
     nStatus = write_data(fpga1_meta, "", (void*) data.byte_data, sizeof(S));

     //close sockets
     close_stream(fpga1_meta);
   
     return 0;  
}
