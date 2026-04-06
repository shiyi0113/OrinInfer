#include "weight_store.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

WeightStore::~WeightStore() {
    if (d_managed_pool_) cudaFree(d_managed_pool_);
}

void WeightStore::load(const ModelLoader& loader, const ModelConfig& config) {
    std::cout << "[WeightStore] Starting Single-Shot Direct IO Loading...\n";

    // 1. 探明整个数据块的物理总长度
    size_t total_data_bytes = 0;
    for (const auto& name : loader.tensor_names()) {
        const auto* t = loader.get_tensor(name);
        size_t end_pos = t->offset + t->byte_size;
        if (end_pos > total_data_bytes) {
            total_data_bytes = end_pos; // 找到最末尾的字节位置
        }
    }

    // 2. 申请唯一的一块连续内存
    CUDA_CHECK(cudaMallocManaged(&d_managed_pool_, total_data_bytes));

    // 3. 终极奥义：仅发起 1 次 pread 系统调用，一口气吞入 1.5GB
    int fd = loader.fd();
    size_t base_file_offset = loader.data_offset();
    
    std::cout << "  -> Executing bulk read of " << (total_data_bytes >> 20) << " MB...\n";
    ssize_t bytes_read = pread(fd, d_managed_pool_, total_data_bytes, base_file_offset);
    if (bytes_read != (ssize_t)total_data_bytes) {
        throw std::runtime_error("Bulk pread failed!");
    }

    // 4. 零成本切分指针：根据每个张量在 Safetensors 里的原生偏移量，直接定位
    auto load_tensor = [&](const std::string& name) -> __nv_bfloat16* {
        auto* t = loader.get_tensor(name);
        if (!t) throw std::runtime_error("[WeightStore] missing tensor: " + name);
        // 直接在基地址上加上原生 offset
        return reinterpret_cast<__nv_bfloat16*>(static_cast<char*>(d_managed_pool_) + t->offset);
    };

    // --- 组装结构体（保持不变） ---
    embedding_ = load_tensor("model.embed_tokens.weight");

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

    final_norm_ = load_tensor("model.norm.weight");
    lm_head_    = load_tensor("lm_head.weight");

    std::cout << "[WeightStore] Load complete.\n";
}