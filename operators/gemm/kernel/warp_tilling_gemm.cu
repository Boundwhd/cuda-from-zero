#include "../gemm.cuh"

constexpr int TILE_SIZE = 32;
constexpr int BLOCKS_ROWS = 8;
constexpr int BLOCKS_COLS = 8;
constexpr int NUM_THREADS = BLOCKS_ROWS * BLOCKS_COLS;
constexpr int WORK_PRE_THREADS_ROWS = TILE_SIZE / BLOCKS_ROWS;
constexpr int WORK_PRE_THREADS_COLS = TILE_SIZE / BLOCKS_COLS;
constexpr int WORK_PER_THREADS = WORK_PRE_THREADS_ROWS * WORK_PRE_THREADS_COLS;

__global__ void gemm_warp_tiling_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) {
    const int tx = threadIdx.x;     
    const int ty = threadIdx.y;

    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;

    __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE][TILE_SIZE];

    const float* global_A_ptr = matrix_A + block_row * K * TILE_SIZE;
    const float* global_B_ptr = matrix_B + block_col * TILE_SIZE;
    float* global_C_ptr = matrix_C + block_row * N * TILE_SIZE + block_col * TILE_SIZE;

    float local_sum[WORK_PRE_THREADS_ROWS][WORK_PRE_THREADS_COLS] = { 0.0f };

    for (int bkidx = 0; bkidx < K; bkidx += TILE_SIZE) {

        for (int i = 0; i < WORK_PER_THREADS; i++) {
            int load_offset = ty * BLOCKS_COLS + tx;
            int r = (load_offset + i * NUM_THREADS) / TILE_SIZE;
            int c = (load_offset + i * NUM_THREADS) % TILE_SIZE;
            smem_A[r][c] = global_A_ptr[r * K + c];
            smem_B[r][c] = global_B_ptr[r * N + c];
        }
        __syncthreads();

        for (int Ty = 0; Ty < WORK_PRE_THREADS_ROWS; Ty++) {
            for (int Tx = 0; Tx < WORK_PRE_THREADS_COLS; Tx++) {
                for (int k = 0; k < TILE_SIZE; k++) {
                    local_sum[Ty][Tx] += smem_A[ty + Ty * BLOCKS_ROWS][k] * smem_B[k][tx + Tx * BLOCKS_COLS];
                }
            }
        }

        global_A_ptr += TILE_SIZE;
        global_B_ptr += TILE_SIZE * N;
    }

    for (int Ty = 0; Ty < WORK_PRE_THREADS_ROWS; Ty++) {
        for (int Tx = 0; Tx < WORK_PRE_THREADS_COLS; Tx++) {
            global_C_ptr[(ty + Ty * BLOCKS_ROWS) * N + (tx + Tx * BLOCKS_COLS)] = local_sum[Ty][Tx];
        }
    }
}

void launch_gemm_warp_tiling_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) { 

    dim3 block_size(BLOCKS_ROWS, BLOCKS_COLS);
    dim3 grid_size(CEIL(M, TILE_SIZE), CEIL(N, TILE_SIZE));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_warp_tiling_f32_f32<<<grid_size, block_size>>>(matrix_A, matrix_B, matrix_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_warp_tiiling_f32_f32 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}