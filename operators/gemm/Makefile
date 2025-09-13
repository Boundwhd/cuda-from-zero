# ----------------------------
# CUDA Makefile for GEMM kernels
# ----------------------------

NVCC := nvcc
CXX := g++

NVCC_FLAGS := -O3 \
              -gencode arch=compute_86,code=sm_86 \
              -std=c++17 \
              -Xcompiler -Wall \
              -lineinfo

SRCS := main.cu $(wildcard kernel/*.cu)
OBJS := $(SRCS:.cu=.o)

TARGET := gemm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean