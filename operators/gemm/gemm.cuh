#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

void launch_gemm_f32_f32_v1(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
);

void launch_gemm_f32_f32_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
);

void launch_gemm_f32_f32_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
);