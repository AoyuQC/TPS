################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

include ./findcudalib.mk

# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda-5.5"

# internal flags
NVCCFLAGS   := -m${OS_SIZE}
CCFLAGS     := -I. -I$(CUDA_PATH)/inlcude `pkg-config --cflags opencv`
NVCCLDFLAGS :=
LDFLAGS     := -L$(CUDA_PATH)/lib64 -lcudart `pkg-config --libs opencv`

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),) 
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH) $(STDLIB)  
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),gnueabi)
      CCFLAGS += -mfloat-abi=softfp
    else
      # default to gnueabihf
      override abi := gnueabihf
      LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
      CCFLAGS += -mfloat-abi=hard
    endif
  endif
endif

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),) 
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
dbg = 1
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      CCFLAGS += -g 
#NVCCFLAGS += -O0
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
#ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(CCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
# INCLUDES  := -I /home/oilpig/cuda-workspace/samples/common/inc
INCLUDES  := -I /home/oilpig/WD/common/inc
LIBRARIES := -L/home/oilpig/Downloads/opencv-2.4.7/release/lib

################################################################################

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

################################################################################

# Target rules
all: build

build: TPSmoduletest 

main.o: main.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_SM30) -o $@ -c $<

tpsCVGPU.o: tpsCVGPU.cpp
	nvcc $(CCFLAGS) $(INCLUDES) -o $@ -c $<	

tpsGPU.o: tpsGPU.cu basisKernel.cuh
	nvcc $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_SM30) -o $@ -c $<

tpsCPU.o: tpsCPU.cpp
	#g++ $(CCFLAGS) $(INCLUDES) -o $@ -c $<
	nvcc -g -G $(CCFLAGS) $(INCLUDES) -o $@ -c $<

TPSmoduletest: main.o tpsCVGPU.o tpsGPU.o tpsCPU.o  
	g++ -g  main.o tpsCVGPU.o tpsCPU.o tpsGPU.o `pkg-config --libs opencv` -L$(CUDA_PATH)/lib64 -o $@ 
	mv $@ ./bin
	#mv main.o tpsCVGPU.o tpsCPU.o tpsGPU.o ./obj 
	

run: build
	./HSOpticalFlow

clean:
	rm -f ./obj/* 
	rm -rf ./bin/TPSmoduletest
clobber: clean
