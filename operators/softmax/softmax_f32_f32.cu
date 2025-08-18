#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
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

// optimize
template<const int warp_per_block = 4>
__global__ void softmax_f32_f32_optimize(float* A, int M, int N, float* B) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int row = blockIdx.x * warp_per_block + warp_id;

    extern __shared__ float s_row[];  // 每个 block 有 warp_per_block 行的空间
    if (row < M) {
        float* s_data = s_row + warp_id * N;  // 每个 warp 用自己的一行

        // 每个 warp 加载自己那一行
        for (int i = lane_id; i < N; i += 32) {
            s_data[i] = expf(A[row * N + i]);
        }
        __syncthreads();  // ✅ 等待所有 warp 加载完成

        // 求和
        float sum = 0.0f;
        for (int i = lane_id; i < N; i += 32) {
            sum += s_data[i];
        }
        for (int offset = 16; offset >= 1; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        // 写出结果
        for (int i = lane_id; i < N; i += 32) {
            B[row * N + i] = s_data[i] / sum;
        }
    }
} 

void launch_softmax_f32_f32_optimize(float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    softmax_f32_f32_optimize<warp_per_block><<<grid_size, block_size, warp_per_block * N, 0>>>(A, M, N, B);

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
        std::cerr << "example: " << argv[0] << " M N" << std::endl;
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

    softmax(A, M, N, B);
    std::cout << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    float* A_D;
    float* B_D;
    cudaMalloc(&A_D, sizeof(float) * M * N);
    cudaMalloc(&B_D, sizeof(float) * M * N);
    
    cudaMemcpy(A_D, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    launch_softmax_f32_f32_naive(A_D, M, N, B_D);
    cudaMemcpy(B, B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    cudaMemset(B_D, 0, sizeof(float) * M * N);
    launch_softmax_f32_f32_optimize(A_D, M, N, B_D);
    cudaMemcpy(B, B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << B[10] << " " << B[500] << " " << B[1000] << std::endl;

    delete(A);
    delete(B);
    cudaFree(A_D);
    cudaFree(B_D);
    return 0;
}