COMMON_REPO = ./
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))


#sw_emu , hw_emu
TARGET := sw_emu

include ./utils.mk

XSA := 
ifneq ($(PLATFORM), )
XSA := $(call device2xsa, $(PLATFORM))
endif

TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

DEVICE := ${AWS_PLATFORM}
XCLBIN := ./xclbin
DSA := $(call device2sandsa, $(DEVICE))

CXX := g++
XOCC := v++

ifneq ($(HOST_ARCH), x86)
    xrt_path = $(SYSROOT)/usr
    OPENCL_INCLUDE:= $(xrt_path)/include/xrt
else
    xrt_path = $(XILINX_XRT)
    OPENCL_INCLUDE:= $(xrt_path)/include
endif
opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++1y

opencl_LDFLAGS=-L$(OPENCL_LIB) -pthread

opencl_LDFLAGS+=-lOpenCL


LDFLAGS += $(opencl_LDFLAGS)

LDFLAGS += -lrt -lstdc++ 

# Kernel compiler global settings
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps -O3
#CLFLAGS +=  --max_memory_ports gs_top
CLFLAGS += -I src/fpga


EXECUTABLE = host

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/gs_top.$(TARGET).$(DSA).xclbin
BINARY_CONTAINER_OBJS += $(XCLBIN)/gs_top.$(TARGET).$(DSA).xo

BINARY_CONTAINERS = $(XCLBIN)/gs_top.$(TARGET).xilinx_pcie-hw-em_7v3_1_0.xclbin
CP = cp -rf

HOST_SRCS = ./src/host.cpp
xcl2_SRCS:=${COMMON_REPO}/libs/xcl2/xcl2.cpp
xcl2_HDRS:=${COMMON_REPO}/libs/xcl2/xcl2.hpp
xcl2_CXXFLAGS:=-I ${COMMON_REPO}/libs/xcl2

CXXFLAGS += ${xcl2_CXXFLAGS}
HOST_SRCS += ${xcl2_SRCS}
HOST_HDRS += ${xcl2_HDRS}


.PHONY: all clean cleanall docs emconfig 
all: $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig run

run: $(EXECUTABLE) $(BINARY_CONTAINERS)
	XCL_EMULATION_MODE=${TARGET} ./$(EXECUTABLE) ${BINARY_CONTAINERS}

.PHONY: exe
exe: emconfig $(EXECUTABLE) $(BINARY_CONTAINERS) run

# Building kernel

$(XCLBIN)/gs_top.$(TARGET).xo: ./src/fpga/gs_top.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k gs_top -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/gs_top.$(TARGET).xilinx_pcie-hw-em_7v3_1_0.xclbin: $(XCLBIN)/gs_top.$(TARGET).xo
	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) -o'$@' $(+)

# Building Host
$(EXECUTABLE): $(HOST_SRCS) $(HOST_HDRS)
	mkdir -p $(XCLBIN)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) ./_x
	-$(RMDIR) ./output.bmp 
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make test PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW are required for SoC shells. Please download and use the pre-built image from - "
	$(ECHO) "      https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html"
	$(ECHO) ""

docs: README.md

README.md: description.json
	$(ABS_COMMON_REPO)/utility/readme_gen/readme_gen.py description.json