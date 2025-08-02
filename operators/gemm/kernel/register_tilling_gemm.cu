#include "../gemm.cuh"

constexpr int TILE_SIZE = 64;
constexpr int BLOCK_ROWS = 16;
constexpr int BLOCK_COLS = 16;
constexpr int NUM_THREADS = BLOCK_COLS * BLOCK_ROWS;
constexpr int WORK_PER_THREADS_ROWS = TILE_SIZE / BLOCK_ROWS;
constexpr int WORK_PER_THREADS_COLS = TILE_SIZE / BLOCK_COLS;
constexpr int WORK_PER_THREADS = (TILE_SIZE * TILE_SIZE) / (BLOCK_ROWS * BLOCK_COLS);


__global__ void gemm_register_tilling_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_col = blockIdx.y;
    const int block_row = blockIdx.x;

    __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE][TILE_SIZE];

    const float* global_A_ptr = matrix_A + block_row * K * TILE_SIZE;
    const float* global_B_ptr = matrix_B + block_col * TILE_SIZE;
    float* global_C_ptr = matrix_C + block_row * N * TILE_SIZE + block_col * TILE_SIZE;

    float local_sum[WORK_PER_THREADS_ROWS][WORK_PER_THREADS_COLS] = { 0.0f };
    // 加一个 float4 的搬运，可以搬运的更快
    // 逻辑:
    // 8 个线程搬运一行
    // 一个 warp 搬运 4 行
    // 两个 warp 搬运 8 行
    // 需要 32 / 8 = 4 躺搬运完毕

    // 加载到 smemA 时进行转置，保证读取 smemA 的时候不会出现 bank conflict
    for (int bkidx = 0; bkidx < K; bkidx += TILE_SIZE) {

        for (int i = 0; i < WORK_PER_THREADS; ++i) {
            int offset = ty * BLOCK_COLS + tx;
            int r = (offset + i * NUM_THREADS) / TILE_SIZE;
            int c = (offset + i * NUM_THREADS) % TILE_SIZE;
            smem_A[r][c] = global_A_ptr[r * K + c];
            smem_B[r][c] = global_B_ptr[r * N + c];
        }
        __syncthreads();

        float reg_A[WORK_PER_THREADS_ROWS] = { 0.0f };
        float reg_B[WORK_PER_THREADS_COLS] = { 0.0f };

        #pragma unroll
        for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx) {
            for (int i = 0; i < WORK_PER_THREADS_ROWS; i++) {
                reg_A[i] = smem_A[ty + i * BLOCK_ROWS][dotIdx];
            }
            
            for (int i = 0; i < WORK_PER_THREADS_COLS; i++) {
                reg_B[i] = smem_B[dotIdx][tx + i * BLOCK_ROWS];
            }

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

void launch_gemm_register_tilling_f32_f32(
    const float* matrix_A,
    const float* matrix_B,
    float* matrix_C,
    const int M,
    const int N,
    const int K
) { 

    dim3 block_size(BLOCK_ROWS, BLOCK_COLS);
    dim3 grid_size(CEIL(M, TILE_SIZE), CEIL(N, TILE_SIZE));
    cudaMemset(matrix_C, 0, sizeof(float) * M * N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gemm_register_tilling_f32_f32<<<grid_size, block_size>>>(matrix_A, matrix_B, matrix_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel gemm_register_tilling_f32_f32 execution time: %.3f ms\n", milliseconds / 10);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}