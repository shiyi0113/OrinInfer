#include "kv_cache.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

// Scatter kernel: for each of the seq_len incoming tokens, compute its
// physical slot via the block table and write K and V there.
//
// Grid : (seq_len,)
// Block: up to 256 threads, each handling a strided slice of kv_dim
__global__ void scatter_kv_kernel(
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    const __nv_bfloat16* __restrict__ k_src,
    const __nv_bfloat16* __restrict__ v_src,
    const int32_t* __restrict__ block_table,
    int start_pos, int seq_len,
    int block_size, int kv_dim
) {
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int logical_pos  = start_pos + t;
    int page         = block_table[logical_pos / block_size];
    int slot         = logical_pos % block_size;
    size_t phys_off  = ((size_t)page * block_size + slot) * kv_dim;

    for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
        k_pool[phys_off + d] = k_src[(size_t)t * kv_dim + d];
        v_pool[phys_off + d] = v_src[(size_t)t * kv_dim + d];
    }
}

// ── KVCache ──────────────────────────────────────────────────────────────────

KVCache::~KVCache() {
    for (auto* p : k_pools_) if (p) cudaFree(p);
    for (auto* p : v_pools_) if (p) cudaFree(p);
    if (d_block_table_) cudaFree(d_block_table_);
}

void KVCache::init(const ModelConfig& config) {
    num_layers_  = config.num_layers;
    kv_dim_      = config.kv_dim();
    total_pages_ = (config.max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Each page holds BLOCK_SIZE tokens; pool covers the full sequence budget.
    size_t pool_bytes = (size_t)total_pages_ * BLOCK_SIZE * kv_dim_
                        * sizeof(__nv_bfloat16);

    k_pools_.resize(num_layers_, nullptr);
    v_pools_.resize(num_layers_, nullptr);
    for (int i = 0; i < num_layers_; i++) {
        CUDA_CHECK(cudaMalloc(&k_pools_[i], pool_bytes));
        CUDA_CHECK(cudaMalloc(&v_pools_[i], pool_bytes));
        CUDA_CHECK(cudaMemset(k_pools_[i], 0, pool_bytes));
        CUDA_CHECK(cudaMemset(v_pools_[i], 0, pool_bytes));
    }

    // Block table: host mirror + device copy
    h_block_table_.assign(total_pages_, -1);
    CUDA_CHECK(cudaMalloc(&d_block_table_, total_pages_ * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_block_table_, -1, total_pages_ * sizeof(int32_t)));

    // Free list: page 0 at the top so it is allocated first
    free_pages_.resize(total_pages_);
    for (int i = 0; i < total_pages_; i++)
        free_pages_[i] = total_pages_ - 1 - i;

    std::cout << "[KVCache] Paged: " << num_layers_ << " layers x "
              << total_pages_ << " pages x " << BLOCK_SIZE << " tok x "
              << kv_dim_ << " dim = "
              << (2LL * num_layers_ * pool_bytes) / (1024 * 1024) << " MB\n";
}

int32_t KVCache::alloc_page() {
    if (free_pages_.empty())
        throw std::runtime_error("[KVCache] Out of physical pages");
    int32_t page = free_pages_.back();
    free_pages_.pop_back();
    return page;
}

void KVCache::prepare(int seq_len, int start_pos) {
    int first_block = start_pos / BLOCK_SIZE;
    int last_block  = (start_pos + seq_len - 1) / BLOCK_SIZE;

    // Allocate physical pages for any logical block that is still unmapped.
    bool updated = false;
    for (int b = first_block; b <= last_block; b++) {
        if (h_block_table_[b] == -1) {
            h_block_table_[b] = alloc_page();
            updated = true;
        }
    }

    if (updated) {
        // Upload only the newly mapped entries to avoid full-table transfers.
        CUDA_CHECK(cudaMemcpy(
            d_block_table_ + first_block,
            h_block_table_.data() + first_block,
            (size_t)(last_block - first_block + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice));
    }
}

void KVCache::scatter_kv(int layer,
                         const __nv_bfloat16* k_src,
                         const __nv_bfloat16* v_src,
                         int seq_len, int start_pos) {
    constexpr int BLOCK = 256;
    scatter_kv_kernel<<<seq_len, BLOCK>>>(
        k_pools_[layer], v_pools_[layer],
        k_src, v_src,
        d_block_table_,
        start_pos, seq_len,
        BLOCK_SIZE, kv_dim_);
}

void KVCache::reset() {
    h_block_table_.assign(total_pages_, -1);
    free_pages_.resize(total_pages_);
    for (int i = 0; i < total_pages_; i++)
        free_pages_[i] = total_pages_ - 1 - i;
    // GPU pool data is stale but will be overwritten before any read.
}
