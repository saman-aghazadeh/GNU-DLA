#ifndef MTU_SIZE
#define MTU_SIZE 1024
#endif

#ifndef TX_CHANNELS
#define TX_CHANNELS 1
#endif

#ifndef RX_CHANNELS
#define RX_CHANNELS 1
#endif

#ifndef ALGO_COPY
#define ALGO_COPY 1
#endif

#ifndef NUM_INPUT_PARMS
#define NUM_INPUT_PARMS 1
#endif

#ifndef NUM_OUTPUT_PARMS
#define NUM_OUTPUT_PARMS 1
#endif

#define SOP 0x1          // Start of Packet
#define EOP 0x2          // End of Packet
#define NULL_PACKET 0x3  // Empty packet
#define HDR_SIZE 0x5

#ifndef RXDATA_LIM_ADDR
#define RXDATA_LIM_ADDR  0*NUM_INPUT_PARMS
#endif
