
# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

CUDA_PATH ?= /usr/local/cuda
SAMPLE_ENABLED := 1
# Common includes and paths for CUDA
INCLUDES  := -I../../../Common
LIBRARIES :=

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

ALL_CCFLAGS += --threads 0 --std=c++11

################################################################################

# Gencode arguments
ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),armv7l aarch64 sbsa))
SMS ?= 53 61 70 72 75 80 86 87 90
else
SMS ?= 50 52 60 61 70 75 80 86 89 90
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ALL_CCFLAGS += --threads 0 --std=c++11

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################
# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif


################################################################################

# Target rules
all: build

build: vectorMAC

vectorMAC.o:vectorMAC.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

vectorMAC: vectorMAC.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	$(EXEC) cp $@ ../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
	$(EXEC) ./vectorMAC

testrun: build

clean:
	rm -f vectorMAC vectorMAC.o
	rm -rf ../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/vectorMAC

clobber: clean