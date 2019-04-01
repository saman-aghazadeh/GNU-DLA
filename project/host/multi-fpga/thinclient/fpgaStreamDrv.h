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

using namespace std;
namespace fpga {

typedef struct _fpga_meta_data{
  string src_mac;
  string dst_mac;
  string type;
  int socket;
  struct sockaddr_ll socket_address;

} fpga_meta;

//must line up with the platform deinition in RX template
typedef enum _region_type {ALGO_CTRL=0, SYS_CTRL, LOAD_CTRL, DATA_CTRL} region_types;

extern int setup_comm(string src_mac, string dst_mac, string tag, fpga_meta * handle);
extern int write_stream_data(fpga_meta handle, string tag, const void * data_in, long size);
extern int write_data(fpga_meta handle, string tag, const void * data_in, long size);
extern int write_ctl(fpga_meta handle, string tag, short int address, int value, bool reset);
extern int write_region(fpga_meta handle, region_types region, unsigned long address, unsigned long value, bool reverse);
extern int read_data(fpga_meta handle, void * data_out, long size_in);
extern void close_stream(fpga_meta handle);
}
