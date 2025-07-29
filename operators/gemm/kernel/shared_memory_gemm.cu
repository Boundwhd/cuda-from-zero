#include "../gemm.cuh"

template<const int BM, const int BN, const int BK>
__global__ void gemm_shared_memory_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) {
    int inner_A_x = threadIdx.x / BK;
    int inner_A_y = threadIdx.x % BK;
    int inner_B_x = threadIdx.x / BN;
    int inner_B_y = threadIdx.x % BN;

    int row = blockIdx.y;
    int col = blockIdx.x;

    const float* A = matrix_A + row * BM * K;
    const float* B = matrix_B + col * BK;
    float* C = matrix_C + row * BM * N + col * BK;
    
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];

    float sum = 0.0f;
    for (int i = 0; i < K; i += BK) {

        smem_A[inner_A_x * BK + inner_A_y] = A[inner_A_x * K + inner_A_y];
        smem_B[inner_B_x * BN + inner_B_y] = B[inner_B_x * N + inner_B_y];

        for (int j = 0; j < BK; j++) {
            sum += smem_A[inner_A_x * BK + j] * smem_B[j * BN + inner_B_y];
        }

        A += BK;
        B += BK * N;
    }
    C[inner_B_x * N + inner_B_y] = sum;
}

void launch_gemm_shared_memory_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) { 
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;

    assert(M % BM == 0);
    assert(N % BN == 0);
    
    dim3 block_size(BM * BN, 1, 1);
    dim3 grid_size(CEIL(N, BM), CEIL(M, BN));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_shared_memory_f32_f32<BM, BN, BK><<<grid_size, block_size>>>(matrix_A, matrix_B, matrix_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_shared_memory_f32_f32 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}