#include "../gemm.cuh"

template <const int BM, const int BN, const int BK, const int NUM_THREADS>
__global__ void gemm_f32_f32_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K    
) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    const float* global_A = A + row * BM * K;
    const float* global_B = B + col * BN;
    float* global_C = C + row * BM * N + col * BN;

    __shared__ float smem_A[BM][BK];
    __shared__ float smem_B[BK][BN];

    constexpr int moves_A = CEIL(BM * BK, NUM_THREADS);
    constexpr int moves_B = CEIL(BK * BN, NUM_THREADS);

    int local_row = threadIdx.x / BN;
    int local_col = threadIdx.x % BN;
    float sum = 0.0f;

    for (int bkidx = 0; bkidx < K; bkidx += BK) {

        // move A to shared memory
        for (int step = 0; step < moves_A; step++) {
            int idx = step * NUM_THREADS + threadIdx.x;
            if (idx < BM * BK) {
                int load_row = idx / BK;
                int load_col = idx % BK;
                if (row * BM + load_row < M && bkidx + load_col < K) {
                    smem_A[load_row][load_col] = global_A[load_row * K + load_col];
                } else {
                    smem_A[load_row][load_col] = 0.0f;
                }
            }
        }

        // move B to shared memory
        for (int step = 0; step < moves_B; step++) {
            int idx = step * NUM_THREADS + threadIdx.x;
            if (idx < BK * BN) {
                int load_row = idx / BN;
                int load_col = idx % BN;
                if (col * BN + load_col < N && bkidx + load_row < K) {
                    smem_B[load_row][load_col] = global_B[load_row * N + load_col];
                } else {
                    smem_B[load_row][load_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            sum += smem_A[local_row][k] * smem_B[k][local_col];
        }

        __syncthreads();

        global_A += BK;
        global_B += BK * N;
    }

    if (row * BM + local_row < M && col * BN + local_col < N) {
        global_C[local_row * N + local_col] = sum;
    }
}

void launch_gemm_f32_f32_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 128;
    constexpr int NUM_THREADS = BM * BN;

    assert(BM * BN <= 1024);
    
    dim3 block_size(BM * BN);
    dim3 grid_size(CEIL(N, BN), CEIL(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_f32_f32_v2<BM, BN, BK, NUM_THREADS><<<grid_size, block_size>>>(A, B, C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_f32_f32_v2 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}