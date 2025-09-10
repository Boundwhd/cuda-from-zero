#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include <random>
#include <chrono>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// ---------------- golden reference ----------------
// --------------------------------------------------
void rmsnorm_cpu(
    const float* __restrict__ A,             // input data   [M, N]
    int M, int N,                            // data shape
    float* __restrict__ B,                   // output data  [M, N]
    const float* __restrict__ weight         // weight data  [N]
) {
    for (int row = 0; row < M; ++row) {
        const float* start = A + row * N;
        float* out_start = B + row * N;

        float value_sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            float value = start[col];
            value_sum += value * value;
        }

        float variable = 1.0f / sqrtf(value_sum / N + 1e-6);

        for (int col = 0; col < N; ++col) {
            out_start[col] = start[col] * variable * weight[col];
        }
    }
}

// ----------------- Cuda Kernels -------------------
// --------------------------------------------------

// per warp compute one row
template<const int warp_per_block = 4>
__global__ void rmsnorm_f32_f32_naive(
    const float* __restrict__ A,             
    int M, int N,                            
    float* __restrict__ B,                   
    const float* __restrict__ weight
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int row = blockIdx.x * warp_per_block + warp_id;

    if (row >= M) return;

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        float value = A[row * N + i];
        sum += value * value;
    }

    for (int offset = 16; offset >= 1; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    float variable = 1.0f / sqrtf((sum / static_cast<float>(N)) + 1e-6);

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = A[row * N + i] * variable * weight[i];
    }
}

void launch_rmsnorm_f32_f32_naive(
    const float* __restrict__ A,             
    int M, int N,                            
    float* __restrict__ B,                   
    const float* __restrict__ weight
) {
    const int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = 32 * warp_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    rmsnorm_f32_f32_naive<<<grid_size, block_size>>>(A, M, N, B, weight);
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
    float* h_weight = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }

    for (int i = 0; i < N; ++i) {
        h_weight[i] = dis(gen);
    }

    // cpu validation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    rmsnorm_cpu(h_A, M, N, h_B, h_weight);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    double milliseconds = duration_cpu.count() / 1000.0;
    printf("CPU Kernel execution time: %.3f ms\n", milliseconds);
    std::cout << h_B[0] << " " << h_B[N] << " " << h_B[2 * N] << std::endl;

    // gpu validation
    float *d_A, *d_B, *d_weight;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, M * N * sizeof(float));
    cudaMalloc(&d_weight, N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, N * sizeof(float), cudaMemcpyHostToDevice);

    // naive kernel
    cudaMemset(d_B, 0, M * N * sizeof(float));
    launch_rmsnorm_f32_f32_naive(d_A, M, N, d_B, d_weight);
    memset(h_B, 0, M * N * sizeof(float));
    cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << h_B[0] << " " << h_B[N] << " " << h_B[2 * N] << std::endl;

    return 0;
}