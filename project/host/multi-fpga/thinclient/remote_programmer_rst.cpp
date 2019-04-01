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

int  MTU_SIZE= 1406;

using namespace std;

FILE* file_handle;
bool add_front_padding=true;
bool pad = false;
bool SOT = true;
int active_line_size=0;
int padd_delta=0;
int data_delta=0;
long file_size=0;

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

//this function is in charge of generating 
//at most line size and returning to main,
//it can return less if it reaches EOF
//returns EOF, either TRUE or FALSE
bool format_stream(string &new_str)
{
 int temp=0;
 //this loop will continue
 while( (active_line_size != MTU_SIZE) &&
        (temp != EOF))
 {
     ///////////////////////////////////////////////////////////////////
     if(add_front_padding) {
       //6B padding
       for(int i=0; i<6; i++)
       {
         //start of tranmission logic 
         ////controls whether to start reconfiguration or not
         if(SOT && (i == 5)){
           new_str += (char) 0x01;
         }
         else
           new_str += (char) 0x00;
       }
       //added two bytes to the stream
       active_line_size += 6; 
       //next iteration goes to data
       padd_delta=0;
       add_front_padding=false;
     }
     ///////////////////////////////////////////////////////////////////
     else if( pad ){
       //keep track of the 12B padding 
       for(int i=0; i < (12-padd_delta); i++){
         //break out of while loop if MTU_SIZE is
         //reached
         if(active_line_size == MTU_SIZE) {padd_delta=i; pad=true; break;}
         else new_str+= (char) 0x00; //1B every iteration
         //add one character to the length
         active_line_size += 1;
         //reset padd_delta
         padd_delta=0; 
         //toggles padding off
         pad=false;
       }//end of for loop
     }
     ///////////////////////////////////////////////////////////////////
     else {
       //keeps track of 4B data insertion
       //from file
       string data_seg="";

       if(SOT){
         data_seg = {0x01,0x00,0x00, 0x00};
         SOT=false;      
         //add one character to the length
         active_line_size += 4; 
         //reset data_delta
         data_delta =0; 
         //toggles padding on 
         pad=true;
       }
       else {
           for(int i=0; i < (4-data_delta); i++){
           //break out of while loop if MTU_SIZE is
           //reached or EOF
             if(active_line_size == MTU_SIZE)
               { data_delta=i; pad=false; break;}
             else {
             temp = getc(file_handle); //1B every iteration
             file_size++;
             if(temp != EOF)
               //new_str += temp;
               if(active_line_size < MTU_SIZE)
                    data_seg += (char) temp;
                    //new_str += (char) active_line_size
             else{
               //happen when eof is reached
               pad=false;
               data_delta=i;
               break;
             }
           }
           //add one character to the length
           active_line_size += 1; 
           //reset data_delta
           data_delta =0; 
           //toggles padding on 
           pad=true;
         } //end of for loop
      } //end of else

      //need to reverse data_seg and append it
      reverse(data_seg.begin(), data_seg.end());
      new_str += data_seg;
   } 
     ///////////////////////////////////////////////////////////////////
 } //end of while loop

 //return whether end of file or not 
 if(temp == EOF)
   return 1;
 else 
   return 0;

}

//reference: http://aschauf.landshut.org/fh/linux/udp_vs_raw/ch01s03.html
int main(int argc, const char *argv[])
{
    int sockfd, n;
    int send_result = 0;
    long rst_length=1;
    long iter=1;
    struct sockaddr_ll socket_address;
    string formatted_str; 

    string src_mac({(char)0x14, (char)0x18, (char) 0x77, (char)0x33, (char)0xB0, (char) 0x6B});
    string packet_type({(char)0xAA, (char)0xAA});
    string packet_type_cmd({(char)0xCC, (char)0xCC});
    string fcs({0x00,0x00,0x00,0x00});

    /*other host MAC address*/
    string dest_mac_chan0({(char)0x00, (char)0x0C, (char)0xD7, (char)0x00, (char)0x2C, (char)0x10});
    string dest_mac_chan1({(char)0x00, (char)0x0C, (char)0xD7, (char) 0x00, (char) 0x2C, (char) 0x11});

    /*prepare sockaddr_ll*/
    /*RAW communication*/
    socket_address.sll_family = PF_PACKET;	
    /*we don't use a protocoll above ethernet layer
     *   ->just use anything here*/
    socket_address.sll_protocol = htons(ETH_P_IP);	

    /*index of the network device
     * see full code later how to retrieve it*/
    //em1  =  2
    //em2  =  3
    //p1p2 =  4
    //p2p2 =  5
    socket_address.sll_ifindex  = 3;

    /*ARP hardware identifier is ethernet*/
    socket_address.sll_hatype   = ARPHRD_ETHER;
	
    /*target is another host*/
    socket_address.sll_pkttype  = PACKET_OTHERHOST;

    /*address length*/
    socket_address.sll_halen    = ETH_ALEN;		
    /*MAC - begin*/
    socket_address.sll_addr[0]  = dest_mac_chan0[0];		
    socket_address.sll_addr[1]  = dest_mac_chan0[1];		
    socket_address.sll_addr[2]  = dest_mac_chan0[2];		
    socket_address.sll_addr[3]  = dest_mac_chan0[3];		
    socket_address.sll_addr[4]  = dest_mac_chan0[4];		
    socket_address.sll_addr[5]  = dest_mac_chan0[5];		
    /*MAC - end*/
    socket_address.sll_addr[6]  = 0xAA;/*not used*/
    socket_address.sll_addr[7]  = 0xAA;/*not used*/

    if(argc > 1)
      rst_length = atol(argv[1]);
    if(argc > 2)
      iter = atol(argv[2]);

/*    //Step 0: Init handle .rbf
    if(argc < 2) {cout <<"Not enough arguments"<<endl; return -1;}
    printf("Trying to open:%s\n", argv[1]);
    file_handle = fopen(argv[1], "r");
 
    if(argc > 2) {
      cout <<"Using MTU Size of: "<< argv[2] <<endl; 
      MTU_SIZE=atoi(argv[2]); 
    }

    if(file_handle == NULL) 
        error("ERROR opening file");
*/
    //create header information
    string header="";
           header += dest_mac_chan0;
           header += src_mac;
           header += packet_type;
    string cmd_header="";
           cmd_header += dest_mac_chan0;
           cmd_header += src_mac;
           cmd_header += packet_type_cmd;

    //Step 1: open a socket SOCK_STREAM, SOCK_DGRAM, SOCK_RAW
    //sockfd = socket(AF_PACKET, SOCK_RAW, htons(FPGA_TARGET));
    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd < 0) error("ERROR opening socket");


    //Step 5: reads file stream and formats it
/*    int i=0;
    while(!format_stream(formatted_str)){
     
      //attach payload to header
      string packet = ""; 
             packet += header;
             packet += formatted_str;
             packet += fcs;

      cout << "header size =" << header.length()<<" formatted_str size =" << formatted_str.length()
           << " fcs size =" << fcs.length() << endl; 
      cout <<"sending packet (#"<<i++<<") length="<<packet.length()<<endl;
      send_result = sendto(sockfd,(const void *) packet.c_str(), packet.length(), 0, 
  	        (struct sockaddr*)&socket_address, sizeof(socket_address));
      usleep(100);
      if (send_result == -1) { error("Failed to send data"); }

      //need to erase everything in string and start over
      add_front_padding=true;
      pad = false;
      active_line_size=0;
      padd_delta=0;
      data_delta=0;

      formatted_str.erase(formatted_str.begin(), formatted_str.end());
    }
    //send the last piece of the file
    //since format_stream return EOF at the end
    //the current piece that led to the EOF needs
    //to be written to the FPGA.
    if(formatted_str.length() > 0){

      //attach payload to header
      string packet = ""; 
             packet += header;
             packet += formatted_str;
             packet += fcs;

      cout <<"final packet (#"<<i++<<") length= "<<packet.length()<<endl;
      send_result = sendto(sockfd,(const void *) packet.c_str(), packet.length(), 0, 
  	        (struct sockaddr*)&socket_address, sizeof(socket_address));
      if (send_result == -1) { error("Failed to send data"); }
      //need to erase everything in string and start over
      formatted_str.erase(formatted_str.begin(), formatted_str.end());
    }
*/
    //issue kernel system reset
    for(int j=0; j < iter; j++) {
    string rst_packet = "";
           rst_packet += cmd_header;
           rst_packet += string({0x00, 0x00});
           for(int i=0; i < rst_length; i++){
             rst_packet += 0x80;
             rst_packet += string({0x00,0x00,0x00,0x00,0x00,0x00,0x01});
           }
           rst_packet += fcs; //may need to remove this, maybe!

     cout <<"Sending reset packet"<<endl; 
     send_result = sendto(sockfd,(const void *) rst_packet.c_str(), rst_packet.length(), 0, 
  	        (struct sockaddr*)&socket_address, sizeof(socket_address));
    }     
    //get a line from the file: new size of string would be
    //for every 4bytes in the file add 12B of padding
    //then add 2B in front of the first line ONLY
    cout <<endl<<endl<<"Final data size transmission: "<<file_size<<endl;

    //close file handle
    close(sockfd);
    return 0;
}
