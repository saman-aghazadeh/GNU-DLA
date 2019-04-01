#include <zmq.hpp>
#include <iostream>
#include <queue>
#include <pthread.h>
#include <opencl_drte>

using namespace std;

int main(int argc, char ** argv)
{
  zmq::context_t context(1);
  zmq::socket_t broker(context, ZMQ_ROUTER);
  string drte_resman_id = "ResManager";
  broker.setsockopt (ZMQ_IDENTITY, drte_resman_id , drte_resman_id.length());
  broker.bind("tcp://*:5671")

  //connection to the OpenCL distributed runtime

  return 0;

}
