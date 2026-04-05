#pragma once
#include "config.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ── Tensor descriptor (host-side, mmap'd) ────────────────────
struct TensorInfo {
    std::string name;
    std::string dtype;              // "F16", "F32", "BF16"
    std::vector<int64_t> shape;
    void*    data;                  // host pointer into mmap'd region
    size_t   offset;                // byte offset from tensor data start
    size_t   byte_size;             // total bytes

    size_t num_elements() const;
};

// ── Tokenizer data (parsed from tokenizer.json) ──────────────
struct TokenizerData {
    std::vector<std::string> vocab;             // id → token string
    std::unordered_map<std::string, int32_t> token_to_id;
    // merges as (left_str, right_str) pairs, in priority order
    std::vector<std::pair<std::string, std::string>> merges;

    // special token ids
    int32_t bos_id         = 151643;
    int32_t eos_id         = 151645;
    int32_t im_start_id    = 151644;
    int32_t im_end_id      = 151645;
    int32_t think_start_id = 151667;
    int32_t think_end_id   = 151668;
};

// ── ModelLoader ──────────────────────────────────────────────
// Loads model from HuggingFace-format directory:
//   config.json        → ModelConfig
//   model.safetensors  → weight tensors (mmap)
//   tokenizer.json     → vocab + merges + special tokens
class ModelLoader {
public:
    explicit ModelLoader(const std::string& model_dir);
    ~ModelLoader();

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // ── Access ───────────────────────────────────────────
    const ModelConfig&    config() const { return config_; }
    const TokenizerData&  tokenizer_data() const { return tok_data_; }

    // Get tensor by name (safetensors key)
    // e.g. "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"
    const TensorInfo*     get_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

private:
    // ── Parsing steps ────────────────────────────────────
    void parse_config(const std::string& path);
    void parse_safetensors(const std::string& path);
    void parse_tokenizer(const std::string& path);

    ModelConfig    config_;
    TokenizerData  tok_data_;

    // safetensors mmap
    int     st_fd_        = -1;
    void*   st_mmap_      = nullptr;
    size_t  st_mmap_size_ = 0;
    size_t  st_data_offset_ = 0;   // 8 + header_size

    std::unordered_map<std::string, TensorInfo> tensors_;
};
