#include "kernels.h"
#include <cuda_bf16.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

// Defined in kernels/gemm_cute.cu
// Returns false if M/N/K are not tile-aligned; caller should fallback.
extern bool launch_gemm_cute(__nv_bfloat16* C,
                              const __nv_bfloat16* A,
                              const __nv_bfloat16* B,
                              int M, int N, int K);

// Auto-dispatch: M=1 → hand-written GEMV, M>1 → cuBLAS GEMM
// C[M,N] = A[M,K] * B[N,K]^T

#ifdef USE_CUBLAS
static cublasHandle_t cublas_handle = nullptr;
static void ensure_cublas() {
    if (!cublas_handle) cublasCreate(&cublas_handle);
}
#endif

// GEMV kernel for decode (M=1): computes C[N] = A[K] * B[N,K]^T
__global__ void gemv_kernel(__nv_bfloat16* C, const __nv_bfloat16* A,
                             const __nv_bfloat16* B, int N, int K) {
    int n = blockIdx.x;
    if (n >= N) return;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        sum += __bfloat162float(A[k]) * __bfloat162float(B[n * K + k]);

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) C[n] = __float2bfloat16(sdata[0]);
}

namespace kernels {
void matmul(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
            int M, int N, int K) {
    if (M == 1) {
        int threads = (K < 256) ? K : 256;
        gemv_kernel<<<N, threads, threads * sizeof(float)>>>(C, A, B, N, K);
    } else {
#ifdef USE_CUBLAS
        ensure_cublas();
        // cublasGemmEx supports BF16 (CUDA_R_16BF).
        // cuBLAS is column-major: to compute row-major C = A * B^T,
        // we pass B (as "A") and A (as "B") with swapped op flags.
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     B, CUDA_R_16BF, K,
                     A, CUDA_R_16BF, K,
                     &beta,
                     C, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
        // Try custom CuTe GEMM (requires tile-aligned M/N/K)
        if (!launch_gemm_cute(C, A, B, M, N, K)) {
            // Fallback for non-aligned dims: per-row GEMV (slow, correctness only)
            for (int m = 0; m < M; m++) {
                int threads = (K < 256) ? K : 256;
                gemv_kernel<<<N, threads, threads * sizeof(float)>>>(
                    C + m * N, A + m * K, B, N, K);
            }
        }
#endif
    }
}
}
