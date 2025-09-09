#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// -------------------- helper --------------------
// -------------------------------------------------
void print_matrix(const float* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (j == N - 1) {
                std::cout << std::setprecision(3) << matrix[i * N + j];
            } else {
                std::cout << std::setprecision(3) << matrix[i * N + j] << " ";
            }
        }
        std::cout << std::endl;
    }
}

// ---------------- golden reference ----------------
// --------------------------------------------------
void transpose_cpu(
    const float* A,     // input matrix
    float* B,           // output matrix
    int M, int N        // data shape
) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            B[col * M + row] = A[row * N + col];
        }
    }
}

// ----------------- Cuda Kernels -------------------
// --------------------------------------------------

__global__ void transpose_f32_f32_naive(
    const float* __restrict__ A, 
    float* __restrict__ B,
    int M, int N
) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ty < M && tx < N) {
        B[tx * M + ty] = A[ty * N + tx];
    }
}

void launch_transpose_f32_f32_naive(
    const float* __restrict__ A, 
    float* __restrict__ B,
    int M, int N
) {
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL(N, 32), CEIL(M, 32));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    transpose_f32_f32_naive<<<grid_size, block_size>>>(A, B, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive Kernel execution time: %.3f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <const int TILE_SIZE = 32>
__global__ void transpose_f32_f32_SMEM(
    const float* __restrict__ A, 
    float* __restrict__ B,
    int M, int N    
) {
    __shared__ float smem[TILE_SIZE][TILE_SIZE + 1];    // avoid bank conflict

    int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tx = blockIdx.x * TILE_SIZE + threadIdx.x;

    smem[threadIdx.y][threadIdx.x] = A[ty * N + tx];
    __syncthreads();

    int w_ty = blockIdx.x * TILE_SIZE + threadIdx.y;
    int w_tx = blockIdx.x * TILE_SIZE + threadIdx.x;

    B[w_ty * M + w_tx] = smem[threadIdx.x][threadIdx.y];
}

void launch_transpose_f32_f32_SMEM(
    const float* __restrict__ A, 
    float* __restrict__ B,
    int M, int N
) { 
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL(N, 32), CEIL(M, 32));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    transpose_f32_f32_SMEM<<<grid_size, block_size>>>(A, B, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SMEM Kernel execution time: %.3f ms\n", milliseconds);

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

    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }

    // cpu validation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_A, h_B, M, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    double milliseconds = duration_cpu.count() / 1000.0;
    printf("CPU Kernel execution time: %.3f ms\n", milliseconds);\

    // ** if matrix is large, we don't print all the matrix for validation! **
    // printf("CPU Before Transpose: \n");
    // print_matrix(h_A, M, N);
    // printf("CPU After Transpose: \n");
    // print_matrix(h_B, N, M);

    // gpu validation
    float *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // naive kernel
    cudaMemset(d_B, 0, M * N * sizeof(float));
    launch_transpose_f32_f32_naive(d_A, d_B, M, N);
    memset(h_B, 0, M * N * sizeof(float));
    cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("GPU After Transpose: \n");
    // print_matrix(h_B, N, M);

    // SMEM kernel
    cudaMemset(d_B, 0, M * N * sizeof(float));
    launch_transpose_f32_f32_SMEM(d_A, d_B, M, N);
    memset(h_B, 0, M * N * sizeof(float));
    cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("GPU After Transpose: \n");
    // print_matrix(h_B, N, M);

    return 0;
}