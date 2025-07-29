# Makefile for GEMM project
NVCC := nvcc

SRC_DIR := .
KERNEL_DIR := $(SRC_DIR)/kernel
OBJ_DIR := obj

KERNEL_SRC := $(wildcard $(KERNEL_DIR)/*.cu)
KERNEL_OBJ := $(patsubst $(KERNEL_DIR)/%.cu, $(OBJ_DIR)/%.o, $(KERNEL_SRC))
MAIN_OBJ := $(OBJ_DIR)/main.o

TARGET := gemm

NVCC_FLAGS := -O3 -arch=sm_86 -std=c++17

all: $(TARGET)

$(TARGET): $(MAIN_OBJ) $(KERNEL_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(MAIN_OBJ): $(SRC_DIR)/main.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(KERNEL_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) *.o $(TARGET)

.PHONY: all clean
