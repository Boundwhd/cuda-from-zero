#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cassert>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// ---------------- CUDA error check ----------------
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                        \
    }                                                                        \
} while(0)

// ---------------- Golden reference (non-safe) ----------------
void softmax_cpu(const float* A, int M, int N, float* B) {
    for (int i = 0; i < M; ++i) {
        const float* row_A = A + i * N;
        float* row_B = B + i * N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = std::exp(row_A[j]);
            row_B[j] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < N; ++j) row_B[j] *= inv;
    }
}

// ---------------- Kernels ----------------
// per-warp computes one row (global mem, 2-pass)
template<const int warp_per_block = 4>
__global__ void softmax_f32_f32_naive(const float* __restrict__ A, int M, int N,
                                      float* __restrict__ B) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int row = blockIdx.x * warp_per_block + warp_id;
    if (row >= M) return;

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        sum += __expf(A[row * N + i]);
    }
    for (int off = 16; off >= 1; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    float inv = 1.0f / sum;

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = __expf(A[row * N + i]) * inv;
    }
}

// optimize: stage row->smem (if fits). still 2-pass.
template<const int warp_per_block = 4>
__global__ void softmax_f32_f32_optimize(const float* __restrict__ A, int M, int N,
                                         float* __restrict__ B) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int row = blockIdx.x * warp_per_block + warp_id;
    if (row >= M) return;

    extern __shared__ float s_row[]; 
    float* s_this = s_row + warp_id * N;

    for (int i = lane_id; i < N; i += 32) {
        s_this[i] = __expf(A[row * N + i]);
    }
    __syncwarp();

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) sum += s_this[i];
    for (int off = 16; off >= 1; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    float inv = 1.0f / sum;

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = s_this[i] * inv;
    }
}

// ---------------- Launchers ----------------
void launch_softmax_f32_f32_naive(const float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    softmax_f32_f32_naive<warp_per_block><<<grid_size, block_size>>>(A, M, N, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Softmax_f32_f32_naive: " << ms << " ms\n";
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

void launch_softmax_f32_f32_optimize(const float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    size_t shared_mem_size = size_t(warp_per_block) * size_t(N) * sizeof(float);

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    int limit_optin = 0, limit_default = 0;
    cudaDeviceGetAttribute(&limit_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&limit_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    int smem_limit = limit_optin > 0 ? limit_optin : limit_default;

    if ((int)shared_mem_size > smem_limit) {
        std::cout << "[warn] smem needed " << shared_mem_size
                  << " > limit " << smem_limit
                  << " -> fallback to naive\n";
        launch_softmax_f32_f32_naive(A, M, N, B);
        return;
    }
    if (limit_optin > limit_default && (int)shared_mem_size > limit_default) {
        CUDA_CHECK(cudaFuncSetAttribute(
            softmax_f32_f32_optimize<warp_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_mem_size));
    }

    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    softmax_f32_f32_optimize<warp_per_block><<<grid_size, block_size, shared_mem_size>>>(A, M, N, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Softmax_f32_f32_optimize: " << ms << " ms\n";
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

// ---------------- Check helper ----------------
void check_max_abs_diff(const float* ref, const float* y, int size, const char* tag) {
    float max_diff = 0.f;
    for (int i = 0; i < size; ++i) {
        float d = std::fabs(ref[i] - y[i]);
        if (d > max_diff) max_diff = d;
    }
    std::cout << tag << " | Max abs diff: " << max_diff << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "example: " << argv[0] << " 1024 1024\n";
        return 1;
    }
    int M = static_cast<int>(std::strtol(argv[1], nullptr, 10));
    int N = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    if (M <= 0 || N <= 0) {
        std::cerr << "M and N must be positive.\n";
        return 1;
    }

    std::vector<float> A(M * N), B_ref(M * N), B_gpu(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < M * N; ++i) A[i] = dist(gen);

    auto t0 = std::chrono::high_resolution_clock::now();
    softmax_cpu(A.data(), M, N, B_ref.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "CPU Softmax: " << ms_cpu << " ms\n";

    float *A_D = nullptr, *B_D = nullptr;
    CUDA_CHECK(cudaMalloc(&A_D, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&B_D, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(A_D, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));

    // Run naive
    launch_softmax_f32_f32_naive(A_D, M, N, B_D);
    CUDA_CHECK(cudaMemcpy(B_gpu.data(), B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    check_max_abs_diff(B_ref.data(), B_gpu.data(), M * N, "naive");

    // Run optimize (auto-fallback on smem overflow)
    launch_softmax_f32_f32_optimize(A_D, M, N, B_D);
    CUDA_CHECK(cudaMemcpy(B_gpu.data(), B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    check_max_abs_diff(B_ref.data(), B_gpu.data(), M * N, "optimize");

    CUDA_CHECK(cudaFree(A_D));
    CUDA_CHECK(cudaFree(B_D));
    return 0;
}
