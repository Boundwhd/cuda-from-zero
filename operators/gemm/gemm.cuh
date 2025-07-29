#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <assert.h>

#define CEIL(a, b) (a + b - 1) / b

void launch_gemm_naive_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
);

void launch_gemm_shared_memory_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
);

void launch_gemm_warp_tiling_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
);
