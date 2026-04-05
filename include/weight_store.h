#pragma once
#include "config.h"
#include "model_loader.h"
#include <cuda_fp16.h>
#include <vector>

// Per-layer weight pointers (all on GPU, read-only after init)
struct LayerWeights {
    half* input_layernorm;     // [hidden_size]
    half* q_proj;              // [q_dim, hidden_size]
    half* q_norm;              // [head_dim]
    half* k_proj;              // [kv_dim, hidden_size]
    half* k_norm;              // [head_dim]
    half* v_proj;              // [kv_dim, hidden_size]
    half* o_proj;              // [hidden_size, q_dim]
    half* post_attn_norm;      // [hidden_size]
    half* gate_proj;           // [intermediate_size, hidden_size]
    half* up_proj;             // [intermediate_size, hidden_size]
    half* down_proj;           // [hidden_size, intermediate_size]
};

class WeightStore {
public:
    WeightStore() = default;
    ~WeightStore();

    // Load all weights from ModelLoader into GPU (one-time init)
    void load(const ModelLoader& loader, const ModelConfig& config);

    const half* embedding()  const { return embedding_; }
    const half* final_norm() const { return final_norm_; }
    const half* lm_head()    const { return embedding_; }  // tied
    const LayerWeights& layer(int i) const { return layers_[i]; }

private:
    half* alloc_and_copy(const TensorInfo* tensor);

    half* embedding_  = nullptr;
    half* final_norm_ = nullptr;
    std::vector<LayerWeights> layers_;
    std::vector<void*> all_allocs_;
};
