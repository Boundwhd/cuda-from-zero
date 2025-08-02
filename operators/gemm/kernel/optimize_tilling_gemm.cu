#include "../gemm.cuh"

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

constexpr int BLOCK_ROWS = BM / TM;
constexpr int BLOCK_COLS = BN / TN;
constexpr int NUM_THREADS = BLOCK_ROWS * BLOCK_COLS;

__global__ void gemm_optimize_tilling_f32_f32(
    const float*  matrix_A,
    const float*  matrix_B,
    float*  matrix_C,
    const int M,
    const int N,
    const int K
) {
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;

    const int ty = (threadIdx.x / BLOCK_COLS) * TM;
    const int tx = (threadIdx.x % BLOCK_COLS) * TN;

    __shared__  float smem_A[BK * BM];
    __shared__  float smem_B[BK * BN];

    const int a_tile_row = threadIdx.x / (BK / 4);
    const int a_tile_col = threadIdx.x % (BK / 4) * 4;

    const int b_tile_row = threadIdx.x / (BN / 4);
    const int b_tile_col = threadIdx.x % (BN / 4) * 4;

    float ld_a_reg[4] = {0.};

    float a_frag[TM];
    float b_frag[TN];
    float local_sum[TM][TN] = {0.};
    
    const float* global_A_ptr = matrix_A + block_row * K * BM;
    const float* global_B_ptr = matrix_B + block_col * BN;
    float* global_C_ptr = matrix_C + block_row * N * BM + block_col * BN;

    #pragma unroll
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // load globalA to smemA
        reinterpret_cast<float4*>(&ld_a_reg)[0] = reinterpret_cast<const float4*>(&global_A_ptr[a_tile_row * K + a_tile_col])[0];
        smem_A[(a_tile_col + 0) * BM + a_tile_row] = ld_a_reg[0];
        smem_A[(a_tile_col + 1) * BM + a_tile_row] = ld_a_reg[1];
        smem_A[(a_tile_col + 2) * BM + a_tile_row] = ld_a_reg[2];
        smem_A[(a_tile_col + 3) * BM + a_tile_row] = ld_a_reg[3];

        // load globalB to smemB
        reinterpret_cast<float4*>(&smem_B[b_tile_row * BN + b_tile_col])[0] = reinterpret_cast<const float4*>(&global_B_ptr[b_tile_row * N + b_tile_col])[0];
        __syncthreads();

        global_A_ptr += BK;
        global_B_ptr += BK * N;

        for (int i = 0; i < BK; i++) {
            
            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                reinterpret_cast<float4*>(&a_frag[m])[0] = reinterpret_cast<float4*>(&smem_A[i * BM + ty + m])[0];
            }

            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                reinterpret_cast<float4*>(&b_frag[n])[0] = reinterpret_cast<float4*>(&smem_B[i * BN + tx + n])[0];
            }

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    local_sum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
    }

    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = reinterpret_cast<float4*>(&global_C_ptr[(ty + m * N) + tx + n])[0];
            ctmp.x = local_sum[m][n + 0];
            ctmp.y = local_sum[m][n + 1];
            ctmp.z = local_sum[m][n + 2];
            ctmp.w = local_sum[m][n + 3];
            reinterpret_cast<float4*>(&global_C_ptr[(ty + m) * N + tx + n])[0] = ctmp;
        }
    }
}

void launch_gemm_optimize_tilling_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) { 
    dim3 grid_size(CEIL(M, BM), CEIL(N, BN), 1);
    dim3 block_size(NUM_THREADS, 1, 1);
    cudaMemset(matrix_C, 0, sizeof(float) * M * N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_optimize_tilling_f32_f32<<<grid_size, block_size>>>(matrix_A, matrix_B, matrix_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_optimize_tilling_f32_f32 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}