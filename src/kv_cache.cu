#include "kv_cache.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

KVCache::~KVCache() {
    for (auto* p : k_bufs_) if (p) cudaFree(p);
    for (auto* p : v_bufs_) if (p) cudaFree(p);
}

void KVCache::init(const ModelConfig& config) {
    max_seq_len_ = config.max_seq_len;
    kv_dim_ = config.kv_dim();

    size_t buf_bytes = (size_t)max_seq_len_ * kv_dim_ * sizeof(half);

    k_bufs_.resize(config.num_layers);
    v_bufs_.resize(config.num_layers);

    for (int i = 0; i < config.num_layers; i++) {
        CUDA_CHECK(cudaMalloc(&k_bufs_[i], buf_bytes));
        CUDA_CHECK(cudaMalloc(&v_bufs_[i], buf_bytes));
        CUDA_CHECK(cudaMemset(k_bufs_[i], 0, buf_bytes));
        CUDA_CHECK(cudaMemset(v_bufs_[i], 0, buf_bytes));
    }

    std::cout << "[KVCache] Ring buffer: " << config.num_layers << " layers x "
              << max_seq_len_ << " positions x " << kv_dim_ << " dim = "
              << (2 * config.num_layers * buf_bytes) / (1024*1024) << " MB\n";
}
