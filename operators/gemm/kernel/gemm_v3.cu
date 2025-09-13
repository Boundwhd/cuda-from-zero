#include "../gemm.cuh"

constexpr int BM = 64;
constexpr int BK = 32;
constexpr int BN = 64;

constexpr int TM = 16;
constexpr int TN = 16;

constexpr int ELEMENT_PER_ROW = BM / TM;
constexpr int ELEMENT_PER_COL = BN / TN;

constexpr int NUM_THREADS = TM * TN;
constexpr int RESULT_PER_THREADS = (BM * BN) / (NUM_THREADS);

constexpr int moves_A = CEIL((BM * BK), (NUM_THREADS / 4));
constexpr int moves_B = CEIL((BK * BN), (NUM_THREADS / 4));

__global__ void gemm_f32_f32_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K    
) {
    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;

    __shared__ float smem_A[BK][BM];
    __shared__ float smem_B[BK][BN];

    float sum[RESULT_PER_THREADS] = {0.0f};

    int local_row = threadIdx.x / TN;
    int local_col = threadIdx.x % TN;

    for (int bkidx = 0; bkidx < K; bkidx += BK) {
        // move global_A to shared memory A
        // using float4 optimize
        // when moving A, we transpose shared memory to avoid bank conflict

        #pragma unroll
        for (int step = 0; step < moves_A; ++step) {
            int idx = step * NUM_THREADS + threadIdx.x;
            if (idx < (BM * BK) / 4) {
                int global_A_row = row + (idx * 4) / BK;
                int global_A_col_base = bkidx + (idx * 4) % BK;

                float4 data = {0.0f, 0.0f, 0.0f, 0.0f};

                if (global_A_row < M) {
                    const float* global_A_ptr = A + global_A_row * K + global_A_col_base;

                    if (global_A_col_base + 0 < K) {data.x = global_A_ptr[0];}
                    if (global_A_col_base + 1 < K) {data.y = global_A_ptr[1];}
                    if (global_A_col_base + 2 < K) {data.z = global_A_ptr[2];}
                    if (global_A_col_base + 3 < K) {data.w = global_A_ptr[3];}
                }
                smem_A[global_A_col_base - bkidx + 0][global_A_row - row] = data.x;
                smem_A[global_A_col_base - bkidx + 1][global_A_row - row] = data.y;
                smem_A[global_A_col_base - bkidx + 2][global_A_row - row] = data.z;
                smem_A[global_A_col_base - bkidx + 3][global_A_row - row] = data.w;
            } 
        }

        #pragma unroll
        for (int step = 0; step < moves_B; ++step) {
            int idx = step * NUM_THREADS + threadIdx.x;
            if (idx < (BK * BN) / 4) {
                int global_B_row = bkidx + (idx * 4) / BN;
                int global_B_col_base = col + (idx * 4) % BN;

                float4 data = {0.0f, 0.0f, 0.0f, 0.0f};
                if (global_B_row < K) {
                    const float* global_B_ptr = B + global_B_row * N + global_B_col_base;

                    if (global_B_col_base + 0 < N) { data.x = global_B_ptr[0]; }
                    if (global_B_col_base + 1 < N) { data.y = global_B_ptr[1]; }
                    if (global_B_col_base + 2 < N) { data.z = global_B_ptr[2]; }
                    if (global_B_col_base + 3 < N) { data.w = global_B_ptr[3]; }
                }

                smem_B[global_B_row - bkidx][global_B_col_base - col + 0] = data.x;
                smem_B[global_B_row - bkidx][global_B_col_base - col + 1] = data.y;
                smem_B[global_B_row - bkidx][global_B_col_base - col + 2] = data.z;
                smem_B[global_B_row - bkidx][global_B_col_base - col + 3] = data.w;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROW; ++i) {
            #pragma unroll
            for (int j = 0; j < ELEMENT_PER_COL; ++j) {
                float temp_sum = 0.0f;
                int C_local_row = local_row * ELEMENT_PER_ROW + i;
                int C_local_col = local_col * ELEMENT_PER_COL + j;

                for (int k_inner = 0; k_inner < BK; k_inner++) {
                    temp_sum += smem_A[k_inner][C_local_row] * smem_B[k_inner][C_local_col];
                } 

                sum[i * ELEMENT_PER_COL + j] += temp_sum;
            }
        }

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROW; ++i) {
            #pragma unroll
            for (int j = 0; j < ELEMENT_PER_COL; ++j) {
                int C_local_row = local_row * ELEMENT_PER_ROW + i;
                int C_local_col = local_col * ELEMENT_PER_COL + j;
                if (row + C_local_row < M && col + C_local_col < N) {
                    C[(row + C_local_row) * N + (col + C_local_col)] = sum[i * ELEMENT_PER_COL + j];
                } 
            }
        }
    }
}

void launch_gemm_f32_f32_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    
    dim3 block_size(NUM_THREADS);
    dim3 grid_size(CEIL(N, BN), CEIL(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_f32_f32_v3<<<grid_size, block_size>>>(A, B, C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_f32_f32_v3 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}