#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <float.h>

/**
 * A = [M, N] 
 * do safe-softmax with dim=-1
 * 1. find max
 * 2. compute sum
 * 3. compute result and write back
 */

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// golden reference
void safe_softmax(float* A, int M, int N, float* B) {
    for (int i = 0; i < M; ++i) {
        const float* row_A = A + i * N;
        float* row_B = B + i * N;

        float max_val = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            if (row_A[j] > max_val) {
                max_val = row_A[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float exp_val = std::exp(row_A[j] - max_val);
            row_B[j] = exp_val;
            sum += exp_val;
        }

        for (int j = 0; j < N; ++j) {
            row_B[j] /= sum;
        }
    }
}

// per warp compute one row
template<const int warp_per_block = 4>
__global__ void safe_softmax_f32_f32_naive(float* A, int M, int N, float* B) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int row = blockIdx.x * warp_per_block + warp_id;

    float v_max = -FLT_MAX;
    for (int i = lane_id; i < N; i += 32) {
        v_max = max(v_max, A[row * N + i]);
    }
    for (int offset = 16; offset >= 1; offset >>= 1) {
        v_max = max(v_max, __shfl_xor_sync(0xffffffff, v_max, offset));
    }

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        sum += expf(A[row * N + i] - v_max);
    }
    for (int offset = 16; offset >= 1; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = expf(A[row * N + i] - v_max) / sum;
    }
}

void launch_safe_softmax_f32_f32_naive(float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 8;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    safe_softmax_f32_f32_naive<warp_per_block><<<grid_size, block_size>>>(A, M, N, B);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Safe-softmax_f32_f32_naive kernel execution time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// optimize with shared memory
template<const int warp_per_block = 4>
__global__ void safe_softmax_f32_f32_optimize(float* A, int M, int N, float* B) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row = blockIdx.x * warp_per_block + warp_id;

    extern __shared__ float s_row[];

    for (int i = lane_id; i < N; i += 32) {
        s_row[warp_id * N + i] = A[row * N + i];
    }
    
    float v_max = -FLT_MAX;
    for (int i = lane_id; i < N; i += 32) {
        v_max = max(v_max, s_row[warp_id * N + i]);
    }

    for (int offset = 16; offset >= 1; offset >>= 1) {
        v_max = max(v_max, __shfl_xor_sync(0xffffffff, v_max, offset));
    }

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        float value = expf(s_row[warp_id * N + i] - v_max); 
        s_row[warp_id * N + i] = value;
        sum += value;
    }

    for (int offset = 16; offset >= 1; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = s_row[warp_id * N + i] / sum;
    }

}

void launch_safe_softmax_f32_f32_optimize(float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 8;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    size_t shared_mem_size = warp_per_block * N * sizeof(float);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    safe_softmax_f32_f32_optimize<warp_per_block><<<grid_size, block_size, shared_mem_size, 0>>>(A, M, N, B);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Safe-softmax_f32_f32_optimize kernel execution time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "example: " << argv[0] << " 1024 1024" << std::endl;
        return 1;
    }
    
    int M = static_cast<int>(std::strtol(argv[1], nullptr, 10));
    int N = static_cast<int>(std::strtol(argv[2], nullptr, 10));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float* A = new float[M * N];
    float* B = new float[M * N];
    for (int i = 0; i < M * N; i++) {
        A[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    safe_softmax(A, M, N, B);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU Softmax execution time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Check result: " << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    float* A_D;
    float* B_D;
    cudaMalloc(&A_D, sizeof(float) * M * N);
    cudaMalloc(&B_D, sizeof(float) * M * N);
    cudaMemcpy(A_D, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    launch_safe_softmax_f32_f32_naive(A_D, M, N, B_D);
    cudaMemcpy(B, B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "Check result: " << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    launch_safe_softmax_f32_f32_optimize(A_D, M, N, B_D);
    cudaMemcpy(B, B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "Check result: " << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    delete(A);
    delete(B);
    cudaFree(A_D);
    cudaFree(B_D);
    return 0;
}