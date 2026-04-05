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
    size_t total_bytes = 0;

    auto load_tensor = [&](const std::string& name) -> __nv_bfloat16* {
        auto* t = loader.get_tensor(name);
        if (!t) throw std::runtime_error("[WeightStore] missing tensor: " + name);
        total_bytes += t->byte_size;
        return alloc_and_copy(t);
    };

    // Embedding
    embedding_ = load_tensor("model.embed_tokens.weight");

    // Per-layer weights (28 layers × 11 tensors = 308)
    layers_.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        auto& L = layers_[i];

        L.input_layernorm = load_tensor(p + ".input_layernorm.weight");
        L.q_proj          = load_tensor(p + ".self_attn.q_proj.weight");
        L.q_norm          = load_tensor(p + ".self_attn.q_norm.weight");
        L.k_proj          = load_tensor(p + ".self_attn.k_proj.weight");
        L.k_norm          = load_tensor(p + ".self_attn.k_norm.weight");
        L.v_proj          = load_tensor(p + ".self_attn.v_proj.weight");
        L.o_proj          = load_tensor(p + ".self_attn.o_proj.weight");
        L.post_attn_norm  = load_tensor(p + ".post_attention_layernorm.weight");
        L.gate_proj       = load_tensor(p + ".mlp.gate_proj.weight");
        L.up_proj         = load_tensor(p + ".mlp.up_proj.weight");
        L.down_proj       = load_tensor(p + ".mlp.down_proj.weight");
    }

    // Final norm + LM head
    final_norm_ = load_tensor("model.norm.weight");
    lm_head_    = load_tensor("lm_head.weight");

    std::cout << "[WeightStore] " << all_allocs_.size() << " tensors, "
              << (total_bytes >> 20) << " MB on GPU\n";
}
