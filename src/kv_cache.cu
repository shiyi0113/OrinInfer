#include "kv_cache.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

// Scatter kernel: routes each incoming token to its physical slot.
//
// Grid : (seq_len,)  — one block per token
// Block: up to 256 threads, each handles a strided slice of kv_dim
//
// Routing rule:
//   logical pos < n_sink   →  phys = pos                (sink, stable)
//   logical pos >= n_sink  →  phys = n_sink + (pos - n_sink) % window_size  (ring)
__global__ void scatter_sink_window_kernel(
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    const __nv_bfloat16* __restrict__ k_src,
    const __nv_bfloat16* __restrict__ v_src,
    int start_pos, int n_sink, int window_size, int kv_dim
) {
    int t   = blockIdx.x;
    int pos = start_pos + t;
    int phys = (pos < n_sink)
               ? pos
               : n_sink + (pos - n_sink) % window_size;

    size_t src_off = (size_t)t    * kv_dim;
    size_t dst_off = (size_t)phys * kv_dim;

    for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
        k_pool[dst_off + d] = k_src[src_off + d];
        v_pool[dst_off + d] = v_src[src_off + d];
    }
}

// ── KVCache ──────────────────────────────────────────────────────────────────

KVCache::~KVCache() {
    for (auto* p : k_pools_) if (p) cudaFree(p);
    for (auto* p : v_pools_) if (p) cudaFree(p);
}

// 2 GB reserved for OS, CUDA runtime, and other system needs.
static constexpr size_t SYSTEM_RESERVE_BYTES = 2ULL * 1024 * 1024 * 1024;

void KVCache::init(const ModelConfig& config) {
    num_layers_ = config.num_layers;
    kv_dim_     = config.kv_dim();
    n_sink_     = config.kv_n_sink;

    // ── Probe available GPU memory ────────────────────────────────────────────
    // Weights and activation pool are already allocated at this point, so
    // cudaMemGetInfo reflects the true remaining headroom.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    if (free_bytes <= SYSTEM_RESERVE_BYTES)
        throw std::runtime_error("[KVCache] Not enough GPU memory for KV cache");

    size_t budget = free_bytes - SYSTEM_RESERVE_BYTES;

    // Each token costs 2 (K+V) * num_layers * kv_dim * sizeof(bf16) bytes.
    size_t bytes_per_token = 2ULL * num_layers_ * kv_dim_ * sizeof(__nv_bfloat16);
    int pool_tokens = static_cast<int>(budget / bytes_per_token);

    if (pool_tokens <= n_sink_)
        throw std::runtime_error("[KVCache] GPU memory too small even for sink tokens");

    window_size_ = pool_tokens - n_sink_;

    // ── Allocate pools ────────────────────────────────────────────────────────
    size_t pool_bytes = (size_t)pool_tokens * kv_dim_ * sizeof(__nv_bfloat16);

    k_pools_.resize(num_layers_, nullptr);
    v_pools_.resize(num_layers_, nullptr);
    for (int i = 0; i < num_layers_; i++) {
        CUDA_CHECK(cudaMalloc(&k_pools_[i], pool_bytes));
        CUDA_CHECK(cudaMalloc(&v_pools_[i], pool_bytes));
    }

    std::cout << "[KVCache] Sink+Window: " << num_layers_ << " layers x "
              << "(" << n_sink_ << " sink + " << window_size_ << " window"
              << " = " << pool_tokens << " tok) x " << kv_dim_ << " dim\n"
              << "[KVCache] Budget: " << budget / (1024*1024) << " MB"
              << " → pool: " << (2LL * num_layers_ * pool_bytes) / (1024*1024) << " MB"
              << " | max prefill: " << pool_tokens << " tokens\n";
}

void KVCache::append_kv(int layer,
                        const __nv_bfloat16* k_src,
                        const __nv_bfloat16* v_src,
                        int seq_len, int start_pos) {
    if (seq_len > n_sink_ + window_size_)
        throw std::runtime_error(
            "[KVCache] seq_len (" + std::to_string(seq_len) +
            ") exceeds kv_n_sink + kv_window_size (" +
            std::to_string(n_sink_ + window_size_) + ")");

    constexpr int BLOCK = 256;
    scatter_sink_window_kernel<<<seq_len, BLOCK>>>(
        k_pools_[layer], v_pools_[layer],
        k_src, v_src,
        start_pos, n_sink_, window_size_, kv_dim_);
}
