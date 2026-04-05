#include "kernels.h"
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

// Auto-dispatch: M=1 → GEMV, M>1 → GEMM (cuBLAS)
// C[M,N] = A[M,K] * B[N,K]^T

#ifdef USE_CUBLAS
static cublasHandle_t cublas_handle = nullptr;
static void ensure_cublas() {
    if (!cublas_handle) cublasCreate(&cublas_handle);
}
#endif

// Simple GEMV kernel for decode (M=1)
__global__ void gemv_kernel(half* C, const half* A, const half* B,
                             int N, int K) {
    // Each block computes one output element C[0, n]
    int n = blockIdx.x;
    if (n >= N) return;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += __half2float(A[k]) * __half2float(B[n * K + k]);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) C[n] = __float2half(sdata[0]);
}

namespace kernels {
void matmul(half* C, const half* A, const half* B, int M, int N, int K) {
    if (M == 1) {
        // Decode path: GEMV
        int threads = (K < 256) ? K : 256;
        gemv_kernel<<<N, threads, threads * sizeof(float)>>>(C, A, B, N, K);
    } else {
#ifdef USE_CUBLAS
        // Prefill path: cuBLAS GEMM
        ensure_cublas();
        // cuBLAS expects column-major, so we compute: C^T = B * A^T
        // which gives us C = A * B^T in row-major
        __half alpha_h = __float2half(1.0f);
        __half beta_h  = __float2half(0.0f);
        cublasHgemm(cublas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha_h,
                     B, K,    // B^T: [N,K] stored row-major = [K,N] col-major
                     A, K,    // A:   [M,K] stored row-major = [K,M] col-major
                     &beta_h,
                     C, N);   // C:   [M,N] stored row-major = [N,M] col-major
#else
        // Fallback: naive GEMM (very slow, for correctness testing only)
        // TODO: Replace with tiled CUDA GEMM kernel
        gemv_kernel<<<N, 256, 256 * sizeof(float)>>>(C, A, B, N, K);
        // WARNING: This only works for M=1, need proper GEMM for M>1
#endif
    }
}
}
