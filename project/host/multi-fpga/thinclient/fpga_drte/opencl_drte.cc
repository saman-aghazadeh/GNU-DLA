#include <opencl_drte.h>
#include <functional>
#include <map>


using namespace std;

map<string, function<int(alloc_request req)> >  g_alloc_actions = 
{
  {"allocate_resource",   opencl_rt::allocate_resource},
  {"deallocate_resource", opencl_rt::deallocate_resource},
  {"modify_allocation",   opencl_rt::modify_resource} /*This wont happen in the first REV */ 
}; 

map<string, function<int(rt_request req)> >  g_rt_actions = 
{
  {"send_data",           opencl_rt::send_data  },
  {"recv_data",           opencl_rt::recv_data  },
  {"set_params",          opencl_rt::set_params  }
}; 

///////////////////////////////////////////////////////////////
//////////////////////////Constructor//////////////////////////
///////////////////////////////////////////////////////////////
opencl_rt::opencl_rt()
{
  //initialize allocation_id
  m_allocation_id = 0;
  // Get the OpenCL platform.
  m_platform = findPlatform("Intel(R) FPGA SDK for OpenCL");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }
  
  //Get device handles
  int num_devices=0;
  cl_device_id * devices = NULL;
  devices = getDevices(m_platform, CL_DEVICE_TPYE_ALL, &num_devices);
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);

  // a context for each device
  cl_context context;
  context = clCreateContext(NULL, num_devices, devices, &oclContextCallback, NULL, &status);
  m_platform_information = vector<accel_entry>(num_devices);
  auto &t = m_platform_information;
 
  int i=0;
  for(auto entry =t.begin(); entry != t.end(); entry++)
  {
    //fill in the m_platform stuff device pointers and contexts
    entry.vacant  = true;
    entry.context = context;
    entry.device  = &devices[i]; 
   
  } 

}
///////////////////////////////////////////////////////////////
////////////////////////Alocation section//////////////////////
///////////////////////////////////////////////////////////////
void opencl_rt::_find_device_by_ids(triple vendor_info, vector<uint> & cand)
{
  auto &t = m_platform_information;
  for(int i=0; i < t.size(); i++)
    if(vendor_info == t[i].vendor_info)
      cand.push(i);
}

ushort opencl_rt::_get_avail_slots(uint device_id)
{
  auto &t = m_current_free_slots;
  ushort cnt=0;
  for(int i=0; i < t.size(); i++)
    if(std::get<0>(t[i]) == device_id)
      cnt++;
  return cnt;   
}

bool opencl_rt::_reserve_slot(uint dev_id)
{
   auto t = m_current_free_slots;

   //find a slot with the 
   for(int i=0; i < t.size(); i++)
   {
     item = t[i];
     if(item.first == dev_id)
     {
       //make triple
       auto element = make_tuple((ulong) m_allocation_id,
                                 (ulong) dev_id, 
                                 (ulong) item.second);
       //remove from free slot list 
       m_current_free_slots.erase(m_current_free_slot.begin()+i-1);  
       //place in the m_allocated_slots
       m_allocated_slots.push_back(element);
       //returning true when success allocation
       return true; 
     }

   }
   //could not allocate slot
   return false;

}

void opencl_rt::_configure_from_existing(vector<ushort> dev_idxs, alloc_request req)
{
    
    //step 1: Round robin slot allocation across
    //        all devices in dev_idx
    uint request_slots_cnt = req.active_slots;

    ushort cnt = 0;
    while(request_slots_cnt)
    {
       cnt = cnt % dev_idxs.size();
       uint id = dev_idxs[cnt]

       if( _get_avail_slots(id) )
       {
         _reserve_slot(i);
         request_slots_cnt--;
       }

       cnt++;
    }
    //step 2: move the slot to the m_allocated_slots list
    //Step 3: maybe place an allocation ID in the FPGA

    //last action to increment allocation_id
    //TODO MOVE THIS INTO THE _reserve_slots function
    m_allocation_id++;
}

//the purpose of this function is to lookup IP
//program FPGA
////update m_platform_information
void opencl_rt::_load_fpga_configuration(ushort dev_idx, alloc_request req)
{
  //find IP based on: 
  auto vendor_info = make_tuple(req.vendor_id, req.prod_id, req.versions_id);
  //get binary .aocx from repo
  string program_name = _get_program_from_repo(req.algo_type, vendor_info, 
                                     req.max_slots, req.datatype);

  //update vendor_information
  _update_vendor_slot_info(dev_idx, vendor_info, req.max_slots); 
  //load bitstream into FPGA
  _load_fpga_bitstream(dev_idx, program_name);
  //boot system components
  _configure_system_components(dev_idx);
  //add new slots to the free slot pool
  _add_slots_to_pool(dev_idx);

}

void opencl_rt::_update_vendor_slot_info(ushort dev_id, triple vendor_info, uint max_slots)
{
  auto &t = m_allocated_devices[dev_id];

  //update vendor info
  t.vendor_info = vendor.info;
  //update max_slots
  t.max_slots = max_slots;
}

//load cl_program
//all oyu get here is the cl_program
void opencl_rt::_load_fpga_bitstream(ushort dev_id, string program_name)
{
  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  auto &t = m_allocated_devices[dev_id];
  cl_device_id device = *t.device;
  cl_context context  = t.context;
  string binary_file = getBoardBinaryFile(program_name, device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  cl_program program = createProgramFromBinary(context,
                                               binary_file.c_str(), 
                                               device, 1);
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");
  
  m_allocated_devices[dev_id].program = program;
  m_allocated_devices[dev_id].program_name = program_name;
  m_allocated_devices[dev_id].vacant = true;
  
}

//kick off all system kernels
//including creating base pointers for all slots
////save it to the m_platform_information
void opencl_rt::configure_system_components(ushort dev_id)
{

  auto &t               = m_allocated_devices[dev_id];
  cl_device_id  device  = t.device;
  cl_program    program = t.program; 
  cl_context    context = t.context;
  auto &cl_obj          = t.cl_obj;
  auto &inputs          = t.inputs;
  auto &outputs         = t.outputs;
  auto get_kernel_obj   = [t] (string s)->cl_kernel
                          {
                            for(auto item : t.cl_obj)
                              if( s.compare(get<0>(item)) == 0)
                                return get<1>(item);  
                          };

  //kernels and queue mapping
  //pair is starting indx and length
  //queue information
  //this implies that the index starts at 0 for set args
  map< string, initializer_list<pair <int> > > sys_kernels {
    {"algo", {{0, 1},
              {2,t.max_slots}}
              {t.max_slots+2,t.max_slots}}
             },
    {"rx_data_arb",{{1,1},
                    {2,t.max_slots} }
             }, 
    {"tx_rdma_arb",{{1,1},
                    {t.max_slots+2,t.max_slots} } 
             }, 
    {"rx_ctrl_arb",{{0,1},
                    {2,1} }
             }, 
    {"rx_io_channel",{{},
                      {} }
             }, 
    {"tx_io_channel"{{},
                     {}  }
             }
  };

  //Create command queues for each kernel
  for(auto kernel : sys_kernels)
  {
    cl_command_queue q = clCreateCommandQueue(context, device, 
                                              CL_QUEUE_PROFILING_ENABLE, 
                                              NULL);
    cl_kernel kern     = clCreateKernel(program, kernel.first, &status);
    //save to class structure
    cl_obj.push_back( make_tuple(kernel.first, kern, q) );
  }
 
  /////////////////ALOCATE ALGO and SYSTEM MEMORY//////////////////////
  for(int i=0; i < 2; i++)
  {
    //setup  algo = 0 and sys = 1
    inputs.push_back( clCreateBuffer(context, CL_MEM_READ_ONLY, 
                        sizeof(unsigned long), NULL, NULL) );
  }

  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////

  //the plus two is for the algo area and system regions
  //placeholder allocations to start the kernels
  for(int i=0; i < t.max_slots; i++)
  {
    //setup input
    inputs.push_back( clCreateBuffer(context, CL_MEM_READ_ONLY, 
                        sizeof(unsigned long), NULL, NULL) );
    //setup output 
    outputs.push_back( clCreateBuffer(context, CL_MEM_READ_ONLY, 
                        sizeof(unsigned long), NULL, NULL) );
  }

  //key buffer allocatoin
  //0,1: 0 = Algo space
  //     1 = System
  //     2+ = RX
  //     max_port+ = TX  
  vector<cl_mem> device_buffers(input.size()+ output.size() );

  device_buffers(device_buffers.end(), input.begin(), input.end());
  device_buffers(device_buffers.end(), output.begin(), output.end());
  //1. set args for the algo kernel
  //Goes through the kernel list
  for(auto kernel : sys_kernels)
  { 
    
    //resets every time kernel changes
    int index = 0;
    //get the intiializer list for range
    //starting from queue
    for(auto i : kernel.second)
      //start the range from first to second
      for(int j=i.first; j < (i.first + i.second); j++)
      {
        clSetKernelArg(get_kernel_obj(kernel.first), index, sizeof(cl_mem), &device_buffers[j]);
        index++;
      }

  }

  //Enqueue Tasks (Algo + system kernels)
  for(auto kernel : sys_kernels)
  {
    printf("\nStarting Kernel:%s \n", kernel.first);
    //command queue for specific kernel
    cl_command_queue q = get<2>(kernel.cl_obj);
    //command queue for kern 
    cl_kernel kern = get<1>(kernel.cl_obj);

    status = clEnqueueTask(q, kern, 0, NULL, NULL);
  }

}
//add slots too pool of avail slots
void opencl_rt::_add_slots_to_pool(ushort dev_idx, ushort max_slots)
{
   auto &t = m_current_free_slots;
  
   for(unsigned short  i=0; i < max_slots; i++)
     t.push_back( make_pair(dev_idx, i) );
  
}

void opencl_rt::_configure_from_new(ushort dev_idx, alloc_request req)
{
     
    //step 1: program FPGA with slot specific FPGA
    _load_fpga_configuration(dev_idx, req);
    //step 2: configure from existing
    _configure_from_existing({dev_idx}, req);
}

bool opencl_rt::_find_first_vacant(ushort & device_idx)
{

  auto &t = m_platform_information;

  for(int i=0; i < t.size(); i++)
    if( t[i].vacant )
    { 
      device_idx =i;
      return true;
    }

  return false;
}

static int opencl_rt::allocate_resource(alloc_request req)
{
  bool alloc_complete=false;
  uint avail_slots = 0;
  vector<uint> v_cand;
  vector< ushort > v_device_indx;
  //Step 1: look for a device(s) matching (includes unalloc
  //        FPGAS) vendor, product, and verion IDs
   triple accel_desc = std::make_tuple (req.vendor_id, 
                         req.prod_id, 
                         req.versions_id);
   //return a list of indices of devices in v_cand
   _find_device_by_ids(accel_desc, v_cand);

  //Step 2: out of the candidate FPGA's select the 
  //        best (based on available slots) 
  //        canidate for this request 
  for(uint dev_idx=v_cand.begin(); dev_indx != v_cand.end(); dev_idx++)
  {
    //go through the candidate and check to see if any
    //of them have enough slots to cover the request
    //if not, remove them from candidency.
     
    //Step 3: if slots exists on a particular FPGA
    //        move from m_current_free to m_alloc_slota
    ushort num_slots = _get_avail_slots(dev_idx);
    avail_slots += num_slots;
    //track the available slots across all preprogammed FPGAs
    if( num_slots > 0)
      v_device_indx.append( dev_idx );
  }
  //if there are enough slots across multiple fpga 
  //use an algorithm to maximize the number of FPGAs
  //split (fill the smallest fpga to largest)
  if( total_avail_slots >= req.active_slots )
  {
    //reserve slots on dev_idx
    //this function uses the m_allocation_id
    alloc_complete = true;
    //program FPGA and boot the standard cell
    _configure_from_existing(v_avail_devices, req);
  }

  //will only allocate all slots on a single FPGA
  if(!alloc_complete)
  {
    //find a vacant FPGA
    //returns the first FPGA that is available
    if( _find_first_vacant(dev_idx) )
    {
      //configure vacant FPGA
      _configure_from_new(dev_idx, req);
    }
    else
    {
      printf("No Resources available to fill request");
    }
  }
  return _get_last_allocation();
}

unsigned int opencl_rt::_get_last_allocation()
{
  return m_allocation_id-1; 
}

static int opencl_rt::deallocate_resource(alloc_request req)
{
  return 0;
}

static int opencl_rt::modify_allocation(alloc_request req)
{

  return 0;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
////////////////////////Runtime Section////////////////////////
///////////////////////////////////////////////////////////////
static int opencl_rt::send_data(rt_request req)
{
  return 0;
}

static int opencl_rt::recv_data(rt_request req)
{

  return 0;
}

static int opencl_rt::set_params(rt_request req)
{

  return 0;
}

static vector<ushort> opencl_rt::get_allocation_by_ids(uint allocation_id)
{
  return vector<ushort>(10);
}

void opencl_rt::free_allocation(ulong allocation_id)
{

}

vector<ushort> opencl_rt::get_slots_by_alloc_id(ulong allocation_id)
{

  return vector<ushort>(10);
}

void opencl_rt::modify_slots_cnt(ushort device_id, ushort new_slot_cnt)
{

}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
int main(int argc, char ** argv)
{
  
  //table include /deviceID total context/ used contexts / VendorID / ProductID / Version
  map< string, vector<long> > device_table;
  //forst key is deviceID/ allocation_id/ list of reserved contexts
  map< string, map<string, vector<short> > > allocation_table;

  zmq::context_t context(1);
  zmq::socket_t dealer(context, ZMQ_DEALER);
  string drte_opencl_id = "opencl_rt_dealer";
  dealer.setsockopt (ZMQ_IDENTITY, drte_opencl_id , drte_opencl_id.length());
  dealer.connect("tcp://localhost:5671");
  cout << "Hello World" <<endl;
  
  return 0;

}
