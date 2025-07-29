#include "../gemm.cuh"

__global__ void gemm_naive_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += matrix_A[row * K + i] * matrix_B[i * N + col];
        }
        
        matrix_C[row * N + col] = sum;
    }
}


void launch_gemm_naive_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) {
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL(N, 32), CEIL(M, 32));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_naive_f32_f32<<<grid_size, block_size>>>(matrix_A, matrix_B, matrix_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_naive_f32_f32 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}