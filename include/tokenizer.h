#pragma once
#include "model_loader.h"
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    // Initialize from ModelLoader's parsed tokenizer data
    explicit Tokenizer(const TokenizerData& data);

    // Encode text → token ids
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token ids → text
    std::string decode(const std::vector<int32_t>& ids) const;

    // Decode single token id → string piece
    std::string decode_token(int32_t id) const;

    // Apply chat template: wraps user message with <|im_start|>/<|im_end|>
    std::vector<int32_t> apply_chat_template(const std::string& user_msg) const;

    int32_t bos_id()     const { return bos_id_; }
    int32_t eos_id()     const { return eos_id_; }
    int32_t im_end_id()  const { return im_end_id_; }
    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }

private:
    void bpe_encode(std::vector<int32_t>& tokens) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;

    struct MergePair {
        int32_t left, right;
        bool operator==(const MergePair& o) const { return left == o.left && right == o.right; }
    };
    struct MergePairHash {
        size_t operator()(const MergePair& p) const {
            return std::hash<int64_t>()(((int64_t)p.left << 32) | (uint32_t)p.right);
        }
    };
    struct MergeResult {
        int32_t merged_id;
        int32_t priority;
    };
    std::unordered_map<MergePair, MergeResult, MergePairHash> merges_;

    int32_t bos_id_, eos_id_, im_start_id_, im_end_id_;
};
