#include <vector>
#include "gemm.cuh"

#define M 4096
#define N 4096
#define K 4096

/*====== Device data ptr ======*/
template<typename T>
struct device_data {
    T* matrix_A;
    T* matrix_B;
    T* matrix_C;
};

/*====== Generate data with specified dtype ======*/
template<typename T>
device_data<T> generate_data(){
    std::vector<T> data_A(M * K, 1);
    std::vector<T> data_B(K * N, 1);
    std::vector<T> data_C(M * N, 0);

    T* d_A;
    T* d_B;
    T* d_C;
    cudaMalloc(&d_A, M * K * sizeof(T));
    cudaMalloc(&d_B, K * N * sizeof(T));
    cudaMalloc(&d_C, M * N * sizeof(T));

    cudaMemcpy(d_A, data_A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, data_B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, data_C.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

    return {d_A, d_B, d_C};
}

/*====== Result Validation ======*/
template<typename T>
void validate_result(device_data<T>& matrix_data) {
    T* output_data_cpu = (T*)malloc(M * N * sizeof(T));
    cudaMemcpy(output_data_cpu, matrix_data.matrix_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; i++) {
        if (fabs(output_data_cpu[i] - K) > 1e-1) {
            printf("validate fail!");
            free(output_data_cpu);
            exit(EXIT_FAILURE);
        }
    }

    printf("Validation passed.\n");
    free(output_data_cpu);

}
/*====== Main test ======*/
int main() {   
    device_data<float> matrix_data = generate_data<float>();
    
    printf("===== Matrix Multiplication Test =====\n");
    printf("A[%d x %d] * B[%d x %d] -> C[%d x %d]\n", M, N, N, K, M, K);

    // ---------------------------------------------
    launch_gemm_naive_f32_f32(
        matrix_data.matrix_A,
        matrix_data.matrix_B,
        matrix_data.matrix_C,
        M, N, K
    );
    validate_result<float>(matrix_data);

    // ---------------------------------------------
    launch_gemm_shared_memory_f32_f32(
        matrix_data.matrix_A,
        matrix_data.matrix_B,
        matrix_data.matrix_C,
        M, N, K
    );
    validate_result<float>(matrix_data);

    // ---------------------------------------------
    launch_gemm_warp_tiling_f32_f32(
        matrix_data.matrix_A,
        matrix_data.matrix_B,
        matrix_data.matrix_C,
        M, N, K
    );
    validate_result<float>(matrix_data);

    // ---------------------------------------------
    launch_gemm_register_tilling_f32_f32(
        matrix_data.matrix_A,
        matrix_data.matrix_B,
        matrix_data.matrix_C,
        M, N, K
    );
    validate_result<float>(matrix_data);

    // ---------------------------------------------
    launch_gemm_optimize_tilling_f32_f32(
        matrix_data.matrix_A,
        matrix_data.matrix_B,
        matrix_data.matrix_C,
        M, N, K
    );
    validate_result<float>(matrix_data);

    // ---------------------------------------------

    cudaFree(matrix_data.matrix_A);
    cudaFree(matrix_data.matrix_B);
    cudaFree(matrix_data.matrix_C);
    return 0;
}