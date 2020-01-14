#set source and target for host
HOST_EXE  = run.exe
HOST_EXE_V2 = run_v2.exe
HOST_EXE_V3 = run_v3.exe
HOST_EXE_MULTICORECONV = run_multicoreconv.exe
HOST_EXE_CASCADE_EMU  = run_cascade_emu.exe
HOST_EXE_CHANNEL_TEST = run_channel_test.exe
HOST_EXE_CHANNEL_TEST_TX = run_channel_test_tx.exe
HOST_EXE_CHANNEL_TEST_RX = run_channel_test_rx.exe
HOST_EXE_MAC_TEST = run_mac_test.exe
HOST_EXE_CHANNEL_MULTI_FPGA = run_channel_multi_fpga.exe
HOST_EXE_DLA_NOSYS = run_dla_nosys.exe
HOST_EXE_DLA_NOSYS_MULTI_FPGA = run_dla_nosys_multi_fpga.exe
HOST_EXE_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL = run_dla_nosys_multi_fpga_mmrev1_spatial.exe
HOST_EXE_DLA_NOSYS_MMREV1 = run_dla_nosys_mmrev1.exe
HOST_EXE_ONEMACHINE_MULTI_FPGA = run_onemachine_multi_fpga.exe
HOST_EXE_2D_MMPE = run_2d_mmpe.exe
HOST_EXE_2D_MMCPU = run_2d_mmcpu.exe
HOST_EXE_3D_I3D = run_3d_i3d.exe
HOST_EXE_3D_MULTI_FPGA = run_3d_multi_fpga.exe
# HOST_EXE_3D_MMPE = run_3d_mmpe.exe
HOST_SRCS = ./host/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_V2 = ./host/main_v2.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_V3 = ./host/main_v3.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_MULTICORECONV = ./host/main_multicoreconv.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_CASCADE_EMU = ./host/cascade_emu_main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_CHANNEL_TEST = ./host/channel_test_main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_CHANNEL_TEST_TX = ./host/channel_test_main_tx.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_CHANNEL_TEST_RX = ./host/channel_test_main_rx.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_MAC_TEST = ./host/multi-fpga/template_rxtx_ut.cpp ./host/multi-fpga/fpgaStreamDrv.cpp ../common/AOCLUtils/opencl.cpp ../common/AOCLUtils/options.cpp
HOST_SRCS_CHANNEL_MULTI_FPGA = ./host/multi-fpga/channel/multi_fpga.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_DLA_NOSYS = ./host/DLA-nosys/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_DLA_NOSYS_MULTI_FPGA = ./host/DLA-nosys-multi-fpga/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL = ./host/DLA-nosys-multi-fpga-MMRev1-Spatial/main.cpp  ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_DLA_MMREV1 = ./host/DLA-nosys-MMRev1/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_ONEMACHINE_MULTI_FPGA = ./host/multi-fpga/one_machine_multi_fpga.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_2D_MMPE = ./host/2D_MMPE/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_2D_MMCPU = ./host/2D_MMCPU/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
# HOST_SRCS_3D_MMPE = ./host/3D_MMPE/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_3D_I3D = ./host/3D-I3D/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_SRCS_3D_MULTI_FPGA = ./host/3D-multi-fpga/main.cpp ../common/ocl_util.cpp ../common/timer.cpp

HOST_OBJS = $(HOST_SRCS:%.cpp=%.o)
HOST_OBJS_V2 = $(HOST_SRCS_V2:%.cpp=%.o)
HOST_OBJS_V3 = $(HOST_SRCS_V3:%.cpp=%.o)
HOST_OBJS_MULTICORECONV = $(HOST_SRCS_MULTICORECONV:%.cpp=%.o)
HOST_OBJS_CASCADE_EMU = $(HOST_SRCS_CASCADE_EMU:%.cpp=%.o)
HOST_OBJS_CHANNEL_TEST = $(HOST_SRCS_CHANNEL_TEST:%.cpp=%.o)
HOST_OBJS_MAC_TEST = $(HOST_SRCS_MAC_TEST:%.cpp=%.o)
HOST_OBJS_CHANNEL_TEST_TX = $(HOST_SRCS_CHANNEL_TEST_TX:%.cpp=%.o)
HOST_OBJS_CHANNEL_TEST_RX = $(HOST_SRCS_CHANNEL_TEST_RX:%.cpp=%.o)
HOST_OBJS_CHANNEL_MULTI_FPGA = $(HOST_SRCS_CHANNEL_MULTI_FPGA:%.cpp=%.o)
HOST_OBJS_DLA_NOSYS = $(HOST_SRCS_DLA_NOSYS:%.cpp=%.o)
HOST_OBJS_DLA_NOSYS_MULTI_FPGA = $(HOST_SRCS_DLA_NOSYS_MULTI_FPGA:%.cpp=%.o)
HOST_OBJS_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL = $(HOST_SRCS_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL:%.cpp=%.o)
HOST_OBJS_DLA_NOSYS_MMREV1 = $(HOST_SRCS_DLA_MMREV1:%.cpp=%.o)
HOST_OBJS_ONEMACHINE_MULTI_FPGA = $(HOST_SRCS_ONEMACHINE_MULTI_FPGA:%.cpp=%.o)
HOST_OBJS_2D_MMPE = $(HOST_SRCS_2D_MMPE:%.cpp=%.o)
HOST_OBJS_2D_MMCPU = $(HOST_SRCS_2D_MMCPU:%.cpp=%.o)
# HOST_OBJS_3D_MMPE = $(HOST_SRCS_3D_MMPE:%.cpp=%.o)
HOST_OBJS_3D_I3D = $(HOST_SRCS_3D_I3D:%.cpp=%.o)
HOST_OBJS_3D_MULTI_FPGA = $(HOST_SRCS_3D_MULTI_FPGA:%.cpp=%.o)

# MKL Integration
USE_MKL = 1
ifeq ($(USE_MKL), 1)
MKL_PATH = /home/user/sbiookag/intel
MKL_INCS = $(MKL_PATH)/mkl/include
endif

HOST_INCS = ../common
HOST_DEBUG =

#if you are loading images from OpenCV interfaces please set to 1
USE_OPENCV = 0

#set source and target for device
#supported vendors are xilinx, altera, amd
VENDOR = altera
#select the host archecture  x86|arm32
PLATFORM = x86
#optinal flows are hw|hw_emu|sw_emu
FLOW        = sw_emu
ifeq ($(VENDOR),xilinx)
KERNEL_SRCS = ./device/conv_pipe_xilinx.cl
else
KERNEL_SRCS = ./device/conv_pipe.cl
KERNEL_SRCS_CASCADE_EMU = ./device/conv_pipe_cascade_emu.cl
endif
KERNEL_NAME = conv
KERNEL_NAME_CASCADE_EMU = cascade_emu
KERNEL_DEFS =
KERNEL_INCS = 
KERNEL_DEBUG = 1
#additional pragma for xilinx
CU_NUM = 1
MAX_MEM_PORT = 0
MEM_PORT_SIZE = 512

#select the device binary name
ifeq ($(VENDOR),xilinx)
DEV_EXE = $(KERNEL_NAME).xclbin
else ifeq ($(VENDOR),altera)
DEV_EXE = $(KERNEL_NAME).aocx
DEV_EXE_CASCADE_EMU = $(KERNEL_NAME_CASCADE_EMU).aocx
else ifeq ($(VENDOR),amd)
DEV_EXE =
endif

#host compiler options
CROSS-COMPILE = arm-linux-gnueabihf-
ifeq ($(PLATFORM),x86)
CXX := g++
ifeq ($(USE_OPENCV),1)
#add your OpenCV PATH here
OCV_INCLUDES = -I/usr/local/include/
OCV_LIBDIRS = -L/usr/local/lib 
OCV_LIBS =  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
endif
else ifeq ($(PLATFORM),arm32)
CXX := $(CROSS-COMPILE)g++
ifeq ($(USE_OPENCV),1)
#add your cross compile OpenCV PATH here
OCV_INCLUDES = -I/usr/local/opencv-arm/include/
OCV_LIBDIRS = -L/usr/local/opencv-arm/lib 
OCV_LIBS =  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
endif
endif

#select whether use OpenCV or not
ifeq ($(USE_OPENCV),1)
CXXFLAGS = -g -Wall -std=c++11 -DUSE_OPENCV #-DVERBOSE_OUTPUT
else
CXXFLAGS = -g -O0 -Wall -std=c++11 #-DVERBOSE_OUTPUT
endif

#select whether manually launch free-run kernels
ifeq ($(FLOW),sw_emu)
CXXFLAGS += -DSW_EMU
endif

MODEL = VGG16_TEST

ifeq ($(VENDOR),xilinx)
COMP_CONFIG = -I$(XILINX_SDX)/runtime/include/1_2 -I$(HOST_INCS)  -DFPGA_DEVICE -DXILINX
LINK_CONFIG = -L$(XILINX_SDX)/runtime/lib/x86_64 -lxilinxopencl
else ifeq ($(VENDOR),altera)
ifeq ($(USE_MKL), 0)
COMP_CONFIG = $(shell aocl compile-config) -I$(HOST_INCS) -DFPGA_DEVICE -D$(MODEL)
LINK_CONFIG = $(shell aocl link-config) -lrt
else
COMP_CONFIG = $(shell aocl compile-config) -I$(HOST_INCS) -I$(MKL_INCS) -Wl,--start-group "$(MKL_PATH)/mkl/lib/intel64"/libmkl_intel_lp64.a "$(MKL_PATH)/mkl/lib/intel64"/libmkl_intel_thread.a "$(MKL_PATH)/mkl/lib/intel64"/libmkl_core.a -Wl,--end-group -DFPGA_DEVICE -D$(MODEL)
LINK_CONFIG = $(shell aocl link-config) -lrt -L"$(MKL_PATH)/compilers_and_libraries/linux/lib/intel64" -liomp5 -lpthread -ldl -lm -Wl,--start-group "$(MKL_PATH)/mkl/lib/intel64"/libmkl_intel_lp64.a "$(MKL_PATH)/mkl/lib/intel64"/libmkl_intel_thread.a "$(MKL_PATH)/mkl/lib/intel64"/libmkl_core.a
endif
else ifeq ($(VENDOR),amd)
DEV_EXE =
endif

#opencl compiler options
#xilinx
ifeq ($(VENDOR),xilinx)
OCC = $(XILINX_SDX)/bin/xocc -DXILINX
OCCFLAGS = --platform xilinx:kcu1500:4ddr-xpr:4.0 -t $(FLOW) -O2  --kernel_frequency 200  --save-temps  --report system \
--memory_port_data_width memWrite:32 \
--memory_port_data_width memRead:512 \
--nk memWrite:1  --nk coreConv:1  --nk lrn:1  --nk maxPool:1  --nk memRead:1 \
#--xp misc:map_connect=add.kernel.memRead_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI\
#--xp misc:map_connect=add.kernel.memRead_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI\
#--xp misc:map_connect=add.kernel.memRead_1.M_AXI_GMEM2.core.OCL_REGION_0.M02_AXI\
#--xp misc:map_connect=add.kernel.memWrite_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI\
#--xp misc:map_connect=add.kernel.lrn_1.M_AXI_GMEM.core.OCL_REGION_0.M00_AXI
#--max_memory_ports lrn\
#--xp misc:map_connect=add.kernel.lrn_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI\
#--xp misc:map_connect=add.kernel.lrn_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI
#memory optimization for xilinx
ifeq ($(MAX_MEM_PORT), 1)
OCCFLAGS += --max_memory_ports all
OCCFLAGS += --memory_port_data_width all:$(MEM_PORT_SIZE)
endif
#altera
else ifeq ($(VENDOR),altera)
OCC = aoc
ifeq ($(FLOW),sw_emu)
OCCFLAGS = -v --report -march=emulator -I device/RTL -L device/RTL -l rtl_lib.aoclib 
else ifeq ($(FLOW),hw)
OCCFLAGS = -v --report --profile -I device/RTL -L device/RTL -l rtl_lib.aoclib 
endif
#amd
else ifeq ($(VENDOR),amd)
OCC =
OCCFLAGS =
endif

#debug option
ifeq ($(KERNEL_DEBUG),1)
	OCCFLAGS += -g
endif

.PHONY: all
all: host fpga

.PHONY: host
ifeq ($(FLOW),sw_emu)
host: $(HOST_EXE) $(HOST_EXE_V2) $(HOST_EXE_V3) $(HOST_EXE_MULTICORECONV) $(HOST_EXE_CASCADE_EMU) $(HOST_EXE_CHANNEL_TEST) $(HOST_EXE_CHANNEL_TEST_TX) $(HOST_EXE_CHANNEL_TEST_RX) $(HOST_EXE_CHANNEL_MULTI_FPGA) $(HOST_EXE_DLA_NOSYS) $(HOST_EXE_DLA_NOSYS_MULTI_FPGA) $(HOST_EXE_DLA_NOSYS_MMREV1) $(HOST_EXE_ONEMACHINE_MULTI_FPGA) $(HOST_EXE_2D_MMPE) $(HOST_EXE_2D_MMCPU) $(HOST_EXE_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL)
else
host: $(HOST_EXE) $(HOST_EXE_V2) $(HOST_EXE_V3) $(HOST_EXE_MULTICORECONV) $(HOST_EXE_CHANNEL_TEST) $(HOST_EXE_CHANNEL_TEST_TX) $(HOST_EXE_CHANNEL_TEST_RX) $(HOST_EXE_DLA_NOSYS) $(HOST_EXE_DLA_NOSYS_MULTI_FPGA) $(HOST_EXE_DLA_NOSYS_MMREV1) $(HOST_EXE_ONEMACHINE_MULTI_FPGA) $(HOST_EXE_2D_MMPE) $(HOST_EXE_2D_MMCPU) $(HOST_EXE_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL)
endif

.PHONY: I3D
I3D: $(HOST_EXE_3D_I3D) $(HOST_EXE_3D_MULTI_FPGA)

.PHONY: fpga
ifeq ($(FLOW),sw_emu)
fpga: $(DEV_EXE) $(DEV_EXE_CASCADE_EMU)
else
fpga: $(DEV_EXE) $(HOST_EXE_CHANNEL_MULTI_FPGA)
endif

$(HOST_EXE): $(HOST_OBJS)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_V2): $(HOST_OBJS_V2)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_V2) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_V3): $(HOST_OBJS_V3)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_V3) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_MULTICORECONV): $(HOST_OBJS_MULTICORECONV)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_MULTICORECONV) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_CASCADE_EMU): $(HOST_OBJS_CASCADE_EMU)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_CASCADE_EMU) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_CHANNEL_TEST): $(HOST_OBJS_CHANNEL_TEST)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_CHANNEL_TEST) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_MAC_TEST): $(HOST_OBJS_MAC_TEST)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_MAC_TEST) -o $@ $(LINK_CONFIG) $(OCV_LIBS)
$(HOST_EXE_CHANNEL_TEST_TX): $(HOST_OBJS_CHANNEL_TEST_TX)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_CHANNEL_TEST_TX) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_CHANNEL_TEST_RX): $(HOST_OBJS_CHANNEL_TEST_RX)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_CHANNEL_TEST_RX) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_CHANNEL_MULTI_FPGA): $(HOST_OBJS_CHANNEL_MULTI_FPGA)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_CHANNEL_MULTI_FPGA) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_DLA_NOSYS): $(HOST_OBJS_DLA_NOSYS)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_DLA_NOSYS) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_DLA_NOSYS_MULTI_FPGA): $(HOST_OBJS_DLA_NOSYS_MULTI_FPGA)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_DLA_NOSYS_MULTI_FPGA) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_DLA_NOSYS_MMREV1): $(HOST_OBJS_DLA_NOSYS_MMREV1)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_DLA_NOSYS_MMREV1) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_ONEMACHINE_MULTI_FPGA): $(HOST_OBJS_ONEMACHINE_MULTI_FPGA)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_ONEMACHINE_MULTI_FPGA) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_2D_MMPE): $(HOST_OBJS_2D_MMPE)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_2D_MMPE) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_2D_MMCPU): $(HOST_OBJS_2D_MMCPU)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_2D_MMCPU) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL): $(HOST_OBJS_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_DLA_NOSYS_MULTI_FPGA_MMREV1_SPATIAL) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_3D_I3D): $(HOST_OBJS_3D_I3D)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_3D_I3D) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

# $(HOST_EXE_3D_MMPE): $(HOST_OBJS_3D_MMPE)
#	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_3D_MMPE) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

$(HOST_EXE_3D_MULTI_FPGA): $(HOST_OBJS_3D_MULTI_FPGA)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS_3D_MULTI_FPGA) -o $@ $(LINK_CONFIG) $(OCV_LIBS)

%.o: %.cpp
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(CXXFLAGS) -c $< -o $@ $(COMP_CONFIG) $(OCV_LIBS)

$(DEV_EXE): $(KERNEL_SRCS)
	$(OCC) $(OCCFLAGS) $< -o $@

ifeq ($(FLOW),sw_emu)
$(DEV_EXE_CASCADE_EMU): $(KERNEL_SRCS_CASCADE_EMU)
	$(OCC) $(OCCFLAGS) $< -o $@
endif

.PHONY: clean
clean:
	rm -rf *.xclbin *.rpt *.xtxt *.json *.dir *.aocx *.aoco *.exe xocc* sdaccel* ./device/*.ll ./host/*.o ../common/*.o *.log *.jou *.mon
