#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <float.h>
#include <cassert>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// ---------- CUDA error check ----------
#define CUDA_CHECK(call)                                                  \
do {                                                                      \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e)             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        std::exit(1);                                                     \
    }                                                                     \
} while(0)

// ---------- golden reference ----------
void safe_softmax_cpu(const float* A, int M, int N, float* B) {
    for (int i = 0; i < M; ++i) {
        const float* row_A = A + i * N;
        float* row_B = B + i * N;

        float max_val = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            if (row_A[j] > max_val) max_val = row_A[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = std::exp(row_A[j] - max_val);
            row_B[j] = e;
            sum += e;
        }

        float inv = 1.0f / sum;
        for (int j = 0; j < N; ++j) row_B[j] *= inv;
    }
}

// ---------- kernels ----------
// per-warp compute one row (naive, 2-pass, global mem)
template<const int warp_per_block = 4>
__global__ void safe_softmax_f32_f32_naive(const float* __restrict__ A, int M, int N,
                                           float* __restrict__ B) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * warp_per_block + warp_id;
    if (row >= M) return;

    float v_max = -FLT_MAX;
    for (int i = lane_id; i < N; i += 32) {
        v_max = fmaxf(v_max, A[row * N + i]);
    }
    for (int off = 16; off >= 1; off >>= 1) {
        v_max = fmaxf(v_max, __shfl_xor_sync(0xffffffff, v_max, off));
    }

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        sum += __expf(A[row * N + i] - v_max);
    }
    for (int off = 16; off >= 1; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }

    float inv = 1.0f / sum;
    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = __expf(A[row * N + i] - v_max) * inv;
    }
}

// optimize: stage row into shared memory (if fits). still 2-pass.
template<const int warp_per_block = 4>
__global__ void safe_softmax_f32_f32_optimize(const float* __restrict__ A, int M, int N,
                                              float* __restrict__ B) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row = blockIdx.x * warp_per_block + warp_id;
    if (row >= M) return;

    extern __shared__ float s_row[];
    float* s_this = s_row + warp_id * N;

    for (int i = lane_id; i < N; i += 32) {
        s_this[i] = A[row * N + i];
    }
    __syncwarp();

    float v_max = -FLT_MAX;
    for (int i = lane_id; i < N; i += 32) {
        v_max = fmaxf(v_max, s_this[i]);
    }
    for (int off = 16; off >= 1; off >>= 1) {
        v_max = fmaxf(v_max, __shfl_xor_sync(0xffffffff, v_max, off));
    }

    float sum = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        float e = __expf(s_this[i] - v_max);
        s_this[i] = e;
        sum += e;
    }
    for (int off = 16; off >= 1; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    float inv = 1.0f / sum;

    for (int i = lane_id; i < N; i += 32) {
        B[row * N + i] = s_this[i] * inv;
    }
}

struct MD { float m, d; };

template<const int warp_per_block = 4>
__global__ void safe_softmax_f32_f32_online(const float* __restrict__ A, int M, int N,
                                            float* __restrict__ B) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * warp_per_block + warp_id;
    if (row >= M) return;

    MD md = {-FLT_MAX, 0.0f};

    for (int i = lane_id; i < N; i += 32) {
        float x = A[row * N + i];
        float m_new = fmaxf(md.m, x);
        float d_new = __expf(md.m - m_new) * md.d + __expf(x - m_new);
        md.m = m_new; md.d = d_new;
    }

    for (int off = 16; off >= 1; off >>= 1) {
        float m_peer = __shfl_xor_sync(0xffffffff, md.m, off);
        float d_peer = __shfl_xor_sync(0xffffffff, md.d, off);
        float m_new = fmaxf(md.m, m_peer);
        float d_new = __expf(md.m - m_new) * md.d + __expf(m_peer - m_new) * d_peer;
        md.m = m_new; md.d = d_new;
    }

    float inv_d = 1.0f / md.d;
    for (int i = lane_id; i < N; i += 32) {
        float x = A[row * N + i];
        B[row * N + i] = __expf(x - md.m) * inv_d;
    }
}

// ---------- launchers ----------
void launch_safe_softmax_f32_f32_naive(const float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 8;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    safe_softmax_f32_f32_naive<warp_per_block><<<grid_size, block_size>>>(A, M, N, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Safe-softmax naive: " << ms << " ms\n";
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

void launch_safe_softmax_f32_f32_optimize(const float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 8;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;
    size_t shared_mem_size = size_t(warp_per_block) * size_t(N) * sizeof(float);

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    int limit1 = 0, limit2 = 0;

    cudaDeviceGetAttribute(&limit1, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&limit2, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    int smem_limit = limit1 > 0 ? limit1 : limit2;

    if ((int)shared_mem_size > smem_limit) {
        std::cout << "[warn] optimize kernel needs " << shared_mem_size
                  << " bytes smem > limit " << smem_limit
                  << " -> fallback to naive.\n";
        launch_safe_softmax_f32_f32_naive(A, M, N, B);
        return;
    }

    if (limit1 > limit2 && (int)shared_mem_size > limit2) {
        CUDA_CHECK(cudaFuncSetAttribute(
            safe_softmax_f32_f32_optimize<warp_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size));
    }

    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    safe_softmax_f32_f32_optimize<warp_per_block><<<grid_size, block_size, shared_mem_size>>>(A, M, N, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Safe-softmax optimize: " << ms << " ms\n";
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

void launch_safe_softmax_f32_f32_online(const float* A, int M, int N, float* B) {
    constexpr int warp_per_block = 4;
    int grid_size = CEIL(M, warp_per_block);
    int block_size = warp_per_block * 32;

    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    safe_softmax_f32_f32_online<warp_per_block><<<grid_size, block_size>>>(A, M, N, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Online safe-softmax: " << ms << " ms\n";
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

// ---------- check helper ----------
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

    std::vector<float> A(M * N), B_ref(M * N), B_gpu(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (int i = 0; i < M * N; ++i) A[i] = dist(gen);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    safe_softmax_cpu(A.data(), M, N, B_ref.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "CPU softmax: " << ms << " ms\n";

    // device buffers
    float *A_D = nullptr, *B_D = nullptr;
    CUDA_CHECK(cudaMalloc(&A_D, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&B_D, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(A_D, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));

    // run naive
    launch_safe_softmax_f32_f32_naive(A_D, M, N, B_D);
    CUDA_CHECK(cudaMemcpy(B_gpu.data(), B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    check_max_abs_diff(B_ref.data(), B_gpu.data(), M * N, "naive");

    // run optimize (auto fallback if smem too big)
    launch_safe_softmax_f32_f32_optimize(A_D, M, N, B_D);
    CUDA_CHECK(cudaMemcpy(B_gpu.data(), B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    check_max_abs_diff(B_ref.data(), B_gpu.data(), M * N, "optimize");

    // run online
    launch_safe_softmax_f32_f32_online(A_D, M, N, B_D);
    CUDA_CHECK(cudaMemcpy(B_gpu.data(), B_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    check_max_abs_diff(B_ref.data(), B_gpu.data(), M * N, "online");

    CUDA_CHECK(cudaFree(A_D));
    CUDA_CHECK(cudaFree(B_D));
    return 0;
}
