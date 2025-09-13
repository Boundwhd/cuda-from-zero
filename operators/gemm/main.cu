#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

#include "gemm.cuh"

// -------------------- HELPER ----------------------
// -------------------------------------------------- 
bool validation(float* cpu_data, float* gpu_data, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(cpu_data[i * N + j] - gpu_data[i * N + j]) > 1e-3f) {
                return false;
            }
        }
    }
    return true;
}

// ---------------- golden reference ----------------
// --------------------------------------------------
void gemm_cpu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s M N K\n", argv[0]);
        return -1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("Running with M = %d, N = %d, K = %d\n", M, N, K);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = dis(gen);
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = dis(gen);
    }

    // cpu validation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_A, h_B, h_C, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    double milliseconds = duration_cpu.count() / 1000.0;
    printf("CPU Kernel execution time: %.3f ms\n", milliseconds);

    // GPU validation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    float* h_valid = (float*)malloc(M * N * sizeof(float));

    cudaMemset(d_C, 0, M * N * sizeof(float));
    launch_gemm_f32_f32_v1(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_valid, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Kernel V1 result: " << std::boolalpha << validation(h_C, h_valid, M, N) << std::endl;

    cudaMemset(d_C, 0, M * N * sizeof(float));
    launch_gemm_f32_f32_v2(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_valid, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Kernel V2 result: " << std::boolalpha << validation(h_C, h_valid, M, N) << std::endl;

    cudaMemset(d_C, 0, M * N * sizeof(float));
    launch_gemm_f32_f32_v3(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_valid, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Kernel V3 result: " << std::boolalpha << validation(h_C, h_valid, M, N) << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_valid);
    return 0;
}