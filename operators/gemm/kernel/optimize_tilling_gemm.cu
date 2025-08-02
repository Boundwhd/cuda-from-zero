#include "../gemm.cuh"

constexpr int TILE_SIZE = 64;
constexpr int BLOCK_ROWS = 16;
constexpr int BLOCK_COLS = 16;
constexpr int NUM_THREADS = BLOCK_COLS * BLOCK_ROWS;
constexpr int WORK_PER_THREADS_ROWS = TILE_SIZE / BLOCK_ROWS;
constexpr int WORK_PER_THREADS_COLS = TILE_SIZE / BLOCK_COLS;
constexpr int WORK_PER_THREADS = (TILE_SIZE * TILE_SIZE) / (BLOCK_ROWS * BLOCK_COLS);

__global__ void gemm_optimize_tilling_f32_f32(
    const float* __restrict__ matrix_A,
    const float* __restrict__ matrix_B,
    float* __restrict__ matrix_C,
    const int M,
    const int N,
    const int K
) {
    const int ty = threadIdx.y;     // thread row   0 ... 16
    const int tx = threadIdx.x;     // thread col   0 ... 16

    const int block_row = blockIdx.y;   // block row
    const int block_col = blockIdx.x;   // block col

    __shared__ float __align__(16) smem_A[TILE_SIZE][TILE_SIZE];  // sram tile A  64 * 64
    __shared__ float __align__(16) smem_B[TILE_SIZE][TILE_SIZE];  // sram tile B  64 * 64

    const float* global_A_ptr = matrix_A + block_row * K * TILE_SIZE;
    const float* global_B_ptr = matrix_B + block_col * TILE_SIZE;
    float* global_C_ptr = matrix_C + block_row * N * TILE_SIZE + block_col * TILE_SIZE;

    float local_sum[WORK_PER_THREADS_ROWS][WORK_PER_THREADS_COLS] = { 0.0f };

    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE) {
        int offset = ty * BLOCK_COLS + tx;  // 0 ... 255
        for (int i = 0; i < WORK_PER_THREADS / 4; ++i) {
            int r = (i * NUM_THREADS + offset) / (TILE_SIZE / 4);
            int c = (i * NUM_THREADS + offset) % (TILE_SIZE / 4);

            float4 tmp_A = reinterpret_cast<const float4*>(&global_A_ptr[r * K + c * 4])[0];
            smem_A[c * 4 + 0][r] = tmp_A.x;
            smem_A[c * 4 + 1][r] = tmp_A.y;
            smem_A[c * 4 + 2][r] = tmp_A.z;
            smem_A[c * 4 + 2][r] = tmp_A.w;

            reinterpret_cast<float4*>(&smem_B[r][c * 4])[0] = reinterpret_cast<const float4*>(&global_B_ptr[r * N + c * 4])[0];
        }
        __syncthreads();

        float reg_A[WORK_PER_THREADS_ROWS] = { 0.0f };
        float reg_B[WORK_PER_THREADS_COLS] = { 0.0f };

        
        for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < WORK_PER_THREADS_ROWS; i++) {
                reg_A[i] = smem_A[dotIdx][ty + i * BLOCK_ROWS];
            }

            #pragma unroll
            for (int i = 0; i < WORK_PER_THREADS_COLS; i++) {
                reg_B[i] = smem_B[dotIdx][tx + i * BLOCK_ROWS];
            }

            #pragma unroll
            for (int i = 0; i < WORK_PER_THREADS_ROWS; i++) {
                for (int j = 0; j < WORK_PER_THREADS_COLS; j++) {
                    local_sum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        global_A_ptr += TILE_SIZE;
        global_B_ptr += TILE_SIZE * N;

    }

    for (int i = 0; i < WORK_PER_THREADS_ROWS; ++i) {
        for (int j = 0; j < WORK_PER_THREADS_COLS; ++j) {
            global_C_ptr[(ty + i * BLOCK_ROWS) * N + (tx + j * BLOCK_COLS)] = local_sum[i][j];
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

    dim3 block_size(BLOCK_ROWS, BLOCK_COLS);
    dim3 grid_size(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));
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