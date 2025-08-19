#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
/**
 * A = [M, N] 
 * do softmax with dim=-1
 */

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// golden reference
void softmax(float* A, int M, int N, float* B) {

    for (int i = 0; i < M; ++i) {
        const float* row_A = A + i * N;
        float* row_B = B + i * N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float exp_val = std::exp(row_A[j]);
            row_B[j] = exp_val;
            sum += exp_val;
        }

        for (int j = 0; j < N; ++j) {
            row_B[j] /= sum;
        }
    }
}

// per warp do one row
template<const int warp_per_block = 4>
__global__ void softmax_f32_f32_naive(float* A, int M, int N, float* B) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int row = blockIdx.x * warp_per_block + warp_id;

    if (row < M) {
        float sum = 0.0f;
        for (int i = lane_id; i < N; i += 32) {
            sum += expf(A[row * N + i]);
        }

        for (int offset = 16; offset >= 1; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        for (int i = lane_id; i < N; i += 32) {
            B[row * N + i] = expf(A[row * N + i]) / sum;
        }
    }
}

void launch_softmax_f32_f32_naive(float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    softmax_f32_f32_naive<warp_per_block><<<grid_size, block_size>>>(A, M, N, B);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Softmax_f32_f32_naive kernel execution time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// optimize with shared memory
template<const int warp_per_block = 4>
__global__ void softmax_f32_f32_optimize(float* A, int M, int N, float* B) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row = blockIdx.x * warp_per_block + warp_id;

    extern __shared__ float s_row[];

    for (int i = lane_id; i < N; i += 32) {
        s_row[warp_id * N + i] = expf(A[row * N + i]);
    }

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        sum += s_row[warp_id * N + i];
    }

    for (int offset = 16; offset >= 1; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = s_row[warp_id * N + i] / sum;
    }
} 

void launch_softmax_f32_f32_optimize(float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    size_t shared_mem_size = warp_per_block * N * sizeof(float);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    softmax_f32_f32_optimize<warp_per_block><<<grid_size, block_size, shared_mem_size, 0>>>(A, M, N, B);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "softmax_f32_f32_optimize kernel execution time: " << elapsed_time << " ms" << std::endl;

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
    softmax(A, M, N, B);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU Softmax execution time: " << duration.count() / 1000.0 << " ms" << std::endl;

    std::cout << "Check result: " << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    float* A_D;
    float* B_D;
    cudaMalloc(&A_D, sizeof(float) * M * N);
    cudaMalloc(&B_D, sizeof(float) * M * N);
    cudaMemcpy(A_D, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    
    launch_softmax_f32_f32_naive(A_D, M, N, B_D);
    cudaMemcpy(B, B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "Check result: " << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    launch_softmax_f32_f32_optimize(A_D, M, N, B_D);
    cudaMemcpy(B, B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "Check result: " << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    delete(A);
    delete(B);
    cudaFree(A_D);
    cudaFree(B_D);
    return 0;
}