#include "kv_cache.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

// KVCache::~KVCache() {
//     for (auto* p : k_bufs_) if (p) cudaFree(p);
//     for (auto* p : v_bufs_) if (p) cudaFree(p);
// }

// void KVCache::init(const ModelConfig& config) {
//     max_seq_len_ = config.max_seq_len;
//     kv_dim_ = config.kv_dim();

//     size_t buf_bytes = (size_t)max_seq_len_ * kv_dim_ * sizeof(__nv_bfloat16);

//     k_bufs_.resize(config.num_layers);
//     v_bufs_.resize(config.num_layers);

//     for (int i = 0; i < config.num_layers; i++) {
//         CUDA_CHECK(cudaMalloc(&k_bufs_[i], buf_bytes));
//         CUDA_CHECK(cudaMalloc(&v_bufs_[i], buf_bytes));
//         CUDA_CHECK(cudaMemset(k_bufs_[i], 0, buf_bytes));
//         CUDA_CHECK(cudaMemset(v_bufs_[i], 0, buf_bytes));
//     }

//     std::cout << "[KVCache] Ring buffer: " << config.num_layers << " layers x "
//               << max_seq_len_ << " positions x " << kv_dim_ << " dim = "
//               << (2 * config.num_layers * buf_bytes) / (1024*1024) << " MB\n";
// }

KVCache::~KVCache() {
    // 析构时只释放这一大块池化内存
    if (pool_base_) cudaFree(pool_base_);
}

void KVCache::init(const ModelConfig& config) {
    max_seq_len_ = config.max_seq_len;
    kv_dim_ = config.kv_dim();

    size_t layer_bytes = (size_t)max_seq_len_ * kv_dim_ * sizeof(__nv_bfloat16);
    // 28层，每层有 K 和 V 两个 buffer
    size_t total_bytes = layer_bytes * config.num_layers * 2; 

    // 绝对不要在循环里 Malloc！一次性全部分配完！
    CUDA_CHECK(cudaMalloc(&pool_base_, total_bytes));
    CUDA_CHECK(cudaMemset(pool_base_, 0, total_bytes));

    k_bufs_.resize(config.num_layers);
    v_bufs_.resize(config.num_layers);

    // 靠算数学偏移量来切割内存
    __nv_bfloat16* base_ptr = static_cast<__nv_bfloat16*>(pool_base_);
    size_t layer_elements = (size_t)max_seq_len_ * kv_dim_;

    for (int i = 0; i < config.num_layers; i++) {
        k_bufs_[i] = base_ptr + (i * 2) * layer_elements;
        v_bufs_[i] = base_ptr + (i * 2 + 1) * layer_elements;
    }

    std::cout << "[KVCache] Ring buffer pool allocated: " 
              << (total_bytes / (1024*1024)) << " MB\n";
}