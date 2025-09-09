#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include <iostream>
#include <random>
#include <chrono>


#define CEIL(a, b) (((a) + (b) - 1) / (b))

// ---------------- golden reference ----------------
// --------------------------------------------------
void layernorm_cpu(
    const float* A,         // input data   [M, N]
    int M, int N,           // data shape
    float* B,               // output data  [M, N]
    const float* scale,     // [N]
    const float* shift      // [N]
) {
    for (int row = 0; row < M; ++row) {
        const float* start = A + row * N;
        float* out_start = B + row * N;

        float mean = 0.0f;
        for (int col = 0; col < N; ++col) {
            mean += start[col];
        }
        mean /= N;

        float var = 0.0f;
        for (int col = 0; col < N; ++col) {
            var += (start[col] - mean) * (start[col] - mean);
        }
        var /= N;

        float inv_std = 1.0 / sqrt(var + 1e-6);

        for (int col = 0; col < N; ++col) {
            float normalized = (start[col] - mean) * inv_std;
            out_start[col] = scale[col] * normalized + shift[col];
        }
    }
}

// ----------------- Cuda Kernels -------------------
// --------------------------------------------------

// per warp compute one row
template<const int warp_per_block = 4>
__global__ void layernorm_f32_f32_naive(
    const float* __restrict__ A,         
    int M, int N,           
    float* __restrict__ B,               
    const float* __restrict__ scale,     
    const float* __restrict__ shift      
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int row = blockIdx.x * warp_per_block + warp_id;

    if (row >= M) {
        return;
    }

    float mean = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        mean += A[row * N + i];
    }

    for (int offset = 16; offset >= 1; offset >>= 1) {
        mean += __shfl_xor_sync(0xffffffff, mean, offset);
    }
    mean /= N;

    float var = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        float diff = A[row * N + i] - mean;
        var += diff * diff;
    }
    
    for (int offset = 16; offset >= 1; offset >>= 1) {
        var += __shfl_xor_sync(0xffffffff, var, offset);
    }
    var /= N;

    float inv_std = 1.0f / sqrtf(var + 1e-6);

    for (int i = lane_id; i < N; i += 32) {
        float normalized = (A[row * N + i] - mean) * inv_std;
        B[row * N + i] = scale[i] * normalized + shift[i];
    }
}

void launch_layernorm_f32_f32_naive(
    const float* __restrict__ A,         
    int M, int N,           
    float* __restrict__ B,               
    const float* __restrict__ scale,     
    const float* __restrict__ shift
) {
    const int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = 32 * warp_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    layernorm_f32_f32_naive<<<grid_size, block_size>>>(A, M, N, B, scale, shift);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive Kernel execution time: %.3f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s M N\n", argv[0]);
        return -1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    printf("Running with M = %d, N = %d\n", M, N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    float* h_A = (float*)malloc(M * N * sizeof(float));
    float* h_B = (float*)malloc(M * N * sizeof(float));
    float* h_scale = (float*)malloc(N * sizeof(float));
    float* h_shift = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }

    for (int i = 0; i < N; ++i) {
        h_scale[i] = dis(gen);
        h_shift[i] = dis(gen);
    }

    // cpu validation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    layernorm_cpu(h_A, M, N, h_B, h_scale, h_shift);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    double milliseconds = duration_cpu.count() / 1000.0;
    printf("CPU Kernel execution time: %.3f ms\n", milliseconds);
    std::cout << h_B[0] << " " << h_B[N] << " " << h_B[2 * N] << std::endl;

    // gpu validation
    float *d_A, *d_B, *d_scale, *d_shift;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, M * N * sizeof(float));
    cudaMalloc(&d_scale, N * sizeof(float));
    cudaMalloc(&d_shift, N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, h_shift, N * sizeof(float), cudaMemcpyHostToDevice);

    // naive kernel
    cudaMemset(d_B, 0, M * N * sizeof(float));
    launch_layernorm_f32_f32_naive(d_A, M, N, d_B, d_scale, d_shift);
    memset(h_B, 0, M * N * sizeof(float));
    cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << h_B[0] << " " << h_B[N] << " " << h_B[2 * N] << std::endl;

    return 0;
}