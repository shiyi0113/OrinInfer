#include "kernels.h"
#include <cfloat>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

// ── softmax ──────────────────────────────────────────────────────────────────

__global__ void softmax_kernel(float* x, int cols) {
    int row = blockIdx.x;
    float* row_ptr = x + row * cols;

    extern __shared__ float sdata[];

    // 1. Find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[i]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(row_ptr[i] - max_val);
        row_ptr[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sdata[0];

    // 3. Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_ptr[i] /= sum;
}

// ── argmax ───────────────────────────────────────────────────────────────────

__global__ void argmax_kernel(int32_t* out, const float* x, int n) {
    __shared__ float  sval[512];
    __shared__ int32_t sidx[512];

    float    local_max = -FLT_MAX;
    int32_t  local_idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > local_max) { local_max = x[i]; local_idx = i; }
    }
    sval[threadIdx.x] = local_max;
    sidx[threadIdx.x] = local_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = sidx[0];
}

// ── element-wise add (in-place: x += y) ──────────────────────────────────────

__global__ void add_kernel(__nv_bfloat16* x, const __nv_bfloat16* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

// ── BF16 → FP32 ──────────────────────────────────────────────────────────────

__global__ void bf16_to_fp32_kernel(float* out, const __nv_bfloat16* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

// ── temperature scaling ───────────────────────────────────────────────────────

__global__ void scale_kernel(float* x, int n, float inv_temp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= inv_temp;
}

// ── namespace kernels ─────────────────────────────────────────────────────────

namespace kernels {

void softmax(float* x, int rows, int cols) {
    int threads = (cols < 1024) ? cols : 1024;
    softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(x, cols);
}

void argmax(int32_t* d_out_token, const float* logits, int vocab_size) {
    argmax_kernel<<<1, 512>>>(d_out_token, logits, vocab_size);
}

void add(__nv_bfloat16* x, const __nv_bfloat16* y, int n) {
    constexpr int THREADS = 256;
    add_kernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(x, y, n);
}

void bf16_to_fp32(float* out, const __nv_bfloat16* in, int n) {
    constexpr int THREADS = 256;
    bf16_to_fp32_kernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(out, in, n);
}

// Top-p (nucleus) sampling with temperature.
// Strategy: apply temperature + softmax on GPU, sort + sample on CPU.
// Acceptable for single-token decode (memory transfer is small relative to generation).
void top_p_sample(int32_t* d_out_token, float* logits, int vocab_size,
                  float temperature, float top_p, unsigned long long seed)
{
    constexpr int THREADS = 256;
    int blocks = (vocab_size + THREADS - 1) / THREADS;

    // 1. Temperature scaling and softmax on GPU
    if (temperature != 1.0f)
        scale_kernel<<<blocks, THREADS>>>(logits, vocab_size, 1.0f / temperature);
    kernels::softmax(logits, 1, vocab_size);
    cudaDeviceSynchronize();

    // 2. Copy probabilities to host
    std::vector<float> h_probs(vocab_size);
    cudaMemcpy(h_probs.data(), logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 3. Sort indices by probability (descending)
    std::vector<int> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return h_probs[a] > h_probs[b];
    });

    // 4. Find top-p cutoff
    int cutoff = vocab_size;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += h_probs[idx[i]];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }

    // 5. Sample from the top-p set
    float norm = 0.0f;
    for (int i = 0; i < cutoff; i++) norm += h_probs[idx[i]];

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, norm);
    float r = dist(rng);

    int32_t token = idx[cutoff - 1];  // fallback to last in top-p
    float cdf = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        cdf += h_probs[idx[i]];
        if (r <= cdf) { token = idx[i]; break; }
    }

    // 6. Write result back to device
    cudaMemcpy(d_out_token, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
}

} // namespace kernels
