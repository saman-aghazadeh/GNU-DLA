#include <zmq.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <pthread.h>
#include <runtime.h>
#include <accelobj.h>
#include <tuple>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace std;

typedef struct _rt_request
{
  //reserve context 0 for localbuffers
  int launching_slot_id;
  int capturing_slot_id;
  int tag;
  long addr;
  void * data;
  vector v_len<int>; 

} rt_request;

//one per FPGA
typedef struct _alloc_request
{
  uint algo_type;        //IP DESIGNATION
  uint input_datatype;   //DATA TYPE INFORMATION 
  uint max_slots; 
  uint active_slots; 
  uint vendor_id;
  uint prod_id;
  uint verions_id;
  uint allocation_id; //used ONLY in case of deallocations and modify 

} alloc_request;

//Opencl accelerator class
public class opencl_accelobj : accelobj
{

};

//opencl main runtime class
//runtime is pure virtual interface
public class opencl_rt 
{
  typedef tuple<uint, uint, uint> triple;

  typedef struct _accel_entry {
    bool vacant;
    cl_context context;
    cl_device_id * device;
    string program_name;
    cl_program  program;
    ushort max_slots;
    triple vendor_info;
    vector< tuple<string, cl_kernel, cl_command_queue> > cl_obj;
    vector<cl_mem> inputs, outputs;

  } accel_entry;

  //default constructor
  opencl_rt(); 
 
  //prupose of this function is to allocate resource 
  //for a single resource. Allocating means calling
  //allocate_resource multiple times and getting
  //multiple allocations IDs
  static int allocate_resource(alloc_request);

  //similar to allocation except deallocate the contexts
  static int deallocate_resource(alloc_request);

  //replaces the semantics of an existing allocation
  //works on a per device and wipes out existing contexts
  static int modify_allocation(alloc_request);

  //used to move data from the broker to the kernel
  static int send_data(rt_request);

  //used to setup the recieve context on the kernels
  static int recv_data(rt_request);

  //used to set system or algo params
  static int set_params(rt_request);

  //get list of unique identifiers for a given allocation
  static vector<ushort> get_allocation_by_ids(uint); 

  private:

  static cl_platform_id m_platform;

  //input allocation id
  //remove allocation from allocated_Slots
  //adds them to the free slot list
  void _free_allocation(ulong);

  //get list of slots given an allocation ID
  vector<ushort> _get_slots_by_alloc_id( ulong );

  //get a list of loaded programs
  vector<string> _get_loaded_programs();

  //get a list of max_slots by device ID
  ushort _get_avail_slots(ushort);
  ushort _get_max_slots(ushort);

  //add slots to pool
  //device_index, number of slots to add
  void _add_slots_to_pool(ushort, ushort);

  //change the max slot count for a specific device
  void _modify_slots_cnt(ushort, ushort); 

  //find a device index by ID
  void _find_device_by_ids(triple, vector<uint> &);
 
  //main configuration function
  void _configure_from_new(ushort, alloc_request);
  void _configure_from_existing(vector<ushort>, alloc_request);

  //reserve slot on the giben dev_idx
  bool _reserve_slot(uint);

  //program specific bitstream to fpga index
  void _load_fpga_bitstream(ushort, string);

  //configure system components
  //extra kernels (inlcuding setting up
  //kernel args and cl_mem
  void _configure_system_components(ushort);

  //updatevendor information and max_slots
  void _update_vendor_slot_info(ushort, triple, ushort);

  //binary grabber from 
  void _get_program_from_repo(uint, triple, uint, uint);

  //get last allocation
  unsigned int _get_last_allocation();

  //MEMBER VARIABLE:

  //keeps track of available slots 
  //device index / slot_number
  vector< pair<ushort, ushort> > m_current_free_slots;
  //keeps a list of allocated slots ( in pairs of allocation_id/device_id/slot_id)
  vector< tuple<ulong, ulong, ulong> > m_allocated_slots;

  //There is a line item for each device
  //the length of this should be the number of accelerators
  //in the system
  //Does not include SetKernelArgs
  vector< alloc_request > m_allocated_devices;
  vector< accel_entry > m_platform_information;
  uint m_allocation_id;
};

