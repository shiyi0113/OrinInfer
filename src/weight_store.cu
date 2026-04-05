#include "weight_store.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

WeightStore::~WeightStore() {
    for (auto* p : all_allocs_) if (p) cudaFree(p);
}

__nv_bfloat16* WeightStore::alloc_and_copy(const TensorInfo* tensor) {
    if (!tensor) throw std::runtime_error("Missing tensor");
    size_t bytes = tensor->byte_size;
    __nv_bfloat16* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, tensor->data, bytes, cudaMemcpyHostToDevice));
    all_allocs_.push_back(d_ptr);
    return d_ptr;
}

void WeightStore::load(const ModelLoader& loader, const ModelConfig& config) {
    std::cout << "[WeightStore] Loading weights to GPU...\n";

    // Embedding
    embedding_ = alloc_and_copy(loader.get_tensor("model.embed_tokens.weight"));

    // Per-layer weights
    layers_.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        auto& L = layers_[i];

        L.input_layernorm = alloc_and_copy(loader.get_tensor(p + ".input_layernorm.weight"));
        L.q_proj          = alloc_and_copy(loader.get_tensor(p + ".self_attn.q_proj.weight"));
        L.q_norm          = alloc_and_copy(loader.get_tensor(p + ".self_attn.q_norm.weight"));
        L.k_proj          = alloc_and_copy(loader.get_tensor(p + ".self_attn.k_proj.weight"));
        L.k_norm          = alloc_and_copy(loader.get_tensor(p + ".self_attn.k_norm.weight"));
        L.v_proj          = alloc_and_copy(loader.get_tensor(p + ".self_attn.v_proj.weight"));
        L.o_proj          = alloc_and_copy(loader.get_tensor(p + ".self_attn.o_proj.weight"));
        L.post_attn_norm  = alloc_and_copy(loader.get_tensor(p + ".post_attention_layernorm.weight"));
        L.gate_proj       = alloc_and_copy(loader.get_tensor(p + ".mlp.gate_proj.weight"));
        L.up_proj         = alloc_and_copy(loader.get_tensor(p + ".mlp.up_proj.weight"));
        L.down_proj       = alloc_and_copy(loader.get_tensor(p + ".mlp.down_proj.weight"));
    }

    // Final norm
    final_norm_ = alloc_and_copy(loader.get_tensor("model.norm.weight"));

    // LM head (separate tensor in safetensors despite tie_word_embeddings=true)
    auto* lmh = loader.get_tensor("lm_head.weight");
    lm_head_  = lmh ? alloc_and_copy(lmh) : embedding_;

    std::cout << "[WeightStore] " << all_allocs_.size() << " tensors loaded to GPU\n";
}
