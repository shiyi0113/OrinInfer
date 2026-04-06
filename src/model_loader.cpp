#include "model_loader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cassert>

// ═══════════════════════════════════════════════════════════════
// Minimal JSON parser (recursive descent, no external dependency)
// Supports: object, array, string, number, bool, null
// ═══════════════════════════════════════════════════════════════

namespace {

struct JsonValue {
    enum Type { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT } type = NUL;
    double                                         num = 0;
    bool                                           boolean = false;
    std::string                                    str;
    std::vector<JsonValue>                         arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    // Object field access
    const JsonValue* get(const std::string& key) const {
        for (auto& [k, v] : obj)
            if (k == key) return &v;
        return nullptr;
    }
    int64_t     as_int()    const { return static_cast<int64_t>(num); }
    float       as_float()  const { return static_cast<float>(num); }
    const std::string& as_str() const { return str; }
    bool        as_bool()   const { return boolean; }
};

static void skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
}

static std::string parse_json_string(const char*& p) {
    assert(*p == '"'); p++;
    std::string s;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"':  s += '"';  break;
                case '\\': s += '\\'; break;
                case '/':  s += '/';  break;
                case 'n':  s += '\n'; break;
                case 't':  s += '\t'; break;
                case 'r':  s += '\r'; break;
                case 'b':  s += '\b'; break;
                case 'f':  s += '\f'; break;
                case 'u': {
                    p++;
                    char hex[5] = {};
                    std::memcpy(hex, p, 4);
                    p += 3; // loop will advance one more
                    uint32_t cp = std::stoul(hex, nullptr, 16);
                    if (cp < 0x80) {
                        s += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        s += static_cast<char>(0xC0 | (cp >> 6));
                        s += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        s += static_cast<char>(0xE0 | (cp >> 12));
                        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        s += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: s += *p; break;
            }
        } else {
            s += *p;
        }
        p++;
    }
    if (*p == '"') p++;
    return s;
}

static JsonValue parse_json_value(const char*& p);

static JsonValue parse_json_value(const char*& p) {
    skip_ws(p);
    JsonValue v;

    if (*p == '"') {
        v.type = JsonValue::STRING;
        v.str = parse_json_string(p);
    } else if (*p == '{') {
        v.type = JsonValue::OBJECT;
        p++; skip_ws(p);
        if (*p != '}') {
            while (true) {
                skip_ws(p);
                std::string key = parse_json_string(p);
                skip_ws(p);
                assert(*p == ':'); p++;
                auto val = parse_json_value(p);
                v.obj.emplace_back(std::move(key), std::move(val));
                skip_ws(p);
                if (*p == ',') { p++; continue; }
                break;
            }
        }
        assert(*p == '}'); p++;
    } else if (*p == '[') {
        v.type = JsonValue::ARRAY;
        p++; skip_ws(p);
        if (*p != ']') {
            while (true) {
                v.arr.push_back(parse_json_value(p));
                skip_ws(p);
                if (*p == ',') { p++; continue; }
                break;
            }
        }
        assert(*p == ']'); p++;
    } else if (*p == 't') {
        v.type = JsonValue::BOOL;
        v.boolean = true;
        p += 4; // "true"
    } else if (*p == 'f') {
        v.type = JsonValue::BOOL;
        v.boolean = false;
        p += 5; // "false"
    } else if (*p == 'n') {
        v.type = JsonValue::NUL;
        p += 4; // "null"
    } else {
        v.type = JsonValue::NUMBER;
        char* end;
        v.num = std::strtod(p, &end);
        p = end;
    }
    return v;
}

static JsonValue parse_json(const std::string& text) {
    const char* p = text.c_str();
    return parse_json_value(p);
}

} // namespace

// ═══════════════════════════════════════════════════════════════
// File helpers
// ═══════════════════════════════════════════════════════════════

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ═══════════════════════════════════════════════════════════════
// TensorInfo
// ═══════════════════════════════════════════════════════════════

size_t TensorInfo::num_elements() const {
    size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ═══════════════════════════════════════════════════════════════
// ModelLoader
// ═══════════════════════════════════════════════════════════════

ModelLoader::ModelLoader(const std::string& model_dir) {
    std::string dir = model_dir;
    if (dir.back() != '/') dir += '/';

    parse_config(dir + "config.json");
    parse_safetensors(dir + "model.safetensors");
    parse_tokenizer(dir + "tokenizer.json");

    std::cout << "[Loader] " << tensors_.size() << " tensors, "
              << tok_data_.vocab.size() << " vocab, "
              << tok_data_.merges.size() << " merges\n";
}

ModelLoader::~ModelLoader() {
    if (st_mmap_ && st_mmap_ != MAP_FAILED) munmap(st_mmap_, st_mmap_size_);
    if (st_fd_ >= 0) close(st_fd_);
}

// ── config.json ──────────────────────────────────────────────

void ModelLoader::parse_config(const std::string& path) {
    auto root = parse_json(read_file(path));

    auto get_int = [&](const char* key, int def) -> int {
        auto* v = root.get(key);
        return v ? static_cast<int>(v->as_int()) : def;
    };
    auto get_float = [&](const char* key, float def) -> float {
        auto* v = root.get(key);
        return v ? v->as_float() : def;
    };
    auto get_bool = [&](const char* key, bool def) -> bool {
        auto* v = root.get(key);
        return v ? v->as_bool() : def;
    };

    config_.vocab_size        = get_int("vocab_size", 151936);
    config_.hidden_size       = get_int("hidden_size", 1024);
    config_.num_layers        = get_int("num_hidden_layers", 28);
    config_.num_heads         = get_int("num_attention_heads", 16);
    config_.num_kv_heads      = get_int("num_key_value_heads", 8);
    config_.head_dim          = get_int("head_dim", 128);
    config_.intermediate_size = get_int("intermediate_size", 3072);
    config_.max_seq_len       = std::min(get_int("max_position_embeddings", 40960), 10240);
    config_.rms_norm_eps      = get_float("rms_norm_eps", 1e-6f);
    config_.rope_theta        = get_float("rope_theta", 1000000.0f);
    config_.tie_embeddings    = get_bool("tie_word_embeddings", true);

    // BOS/EOS from config (Qwen3 defaults)
    config_.bos_id = get_int("bos_token_id", 151643);
    config_.eos_id = get_int("eos_token_id", 151645);

    std::cout << "[Config] layers=" << config_.num_layers
              << " hidden=" << config_.hidden_size
              << " heads=" << config_.num_heads << "/" << config_.num_kv_heads
              << " vocab=" << config_.vocab_size << "\n";
}

// ── model.safetensors ────────────────────────────────────────
// Format:
//   [0:8]      uint64_t header_size (little-endian)
//   [8:8+N]    JSON header string (N = header_size)
//   [8+N:...]  raw tensor data (contiguous, no padding between tensors)
//
// JSON header: { "tensor_name": {"dtype":"F16", "shape":[...], "data_offsets":[start,end]}, ... }
// data_offsets are relative to the start of the tensor data section (byte 8+N)

void ModelLoader::parse_safetensors(const std::string& path) {
    st_fd_ = open(path.c_str(), O_RDONLY);
    if (st_fd_ < 0) throw std::runtime_error("Cannot open: " + path);

    struct stat st;
    fstat(st_fd_, &st);
    st_mmap_size_ = st.st_size;

    st_mmap_ = mmap(nullptr, st_mmap_size_, PROT_READ, MAP_PRIVATE, st_fd_, 0);
    if (st_mmap_ == MAP_FAILED) {
        close(st_fd_);
        throw std::runtime_error("mmap failed: " + path);
    }

    const uint8_t* base = static_cast<const uint8_t*>(st_mmap_);

    // Read header size (8 bytes, little-endian uint64)
    uint64_t header_size = 0;
    std::memcpy(&header_size, base, 8);

    // Parse JSON header
    std::string header_json(reinterpret_cast<const char*>(base + 8), header_size);
    auto root = parse_json(header_json);

    // Tensor data starts after header
    st_data_offset_ = 8 + header_size;

    // Parse each tensor entry
    for (auto& [name, info] : root.obj) {
        // Skip __metadata__ key
        if (name == "__metadata__") continue;

        TensorInfo ti;
        ti.name = name;

        auto* dtype_v = info.get("dtype");
        ti.dtype = dtype_v ? dtype_v->as_str() : "F16";

        auto* shape_v = info.get("shape");
        if (shape_v) {
            for (auto& dim : shape_v->arr)
                ti.shape.push_back(dim.as_int());
        }

        auto* offsets_v = info.get("data_offsets");
        if (offsets_v && offsets_v->arr.size() >= 2) {
            size_t start = static_cast<size_t>(offsets_v->arr[0].num);
            size_t end   = static_cast<size_t>(offsets_v->arr[1].num);
            ti.offset    = start;
            ti.byte_size = end - start;
            ti.data      = const_cast<void*>(
                static_cast<const void*>(base + st_data_offset_ + start));
        }

        tensors_[name] = std::move(ti);
    }

    std::cout << "[Safetensors] " << tensors_.size() << " tensors, "
              << (st_mmap_size_ / (1024 * 1024)) << " MB mapped\n";
}

// ── tokenizer.json ───────────────────────────────────────────
// HuggingFace tokenizer.json structure:
// {
//   "model": {
//     "vocab": { "token_str": id, ... },
//     "merges": [ "left right", ... ]   or  [ ["left","right"], ... ]
//   },
//   "added_tokens": [ {"id": N, "content": "..."}, ... ]
// }

void ModelLoader::parse_tokenizer(const std::string& path) {
    auto root = parse_json(read_file(path));

    auto* model = root.get("model");
    if (!model) throw std::runtime_error("tokenizer.json: missing 'model' key");

    // ── Vocab ────────────────────────────────────────────
    auto* vocab_obj = model->get("vocab");
    if (vocab_obj && vocab_obj->type == JsonValue::OBJECT) {
        // Find max id to size the vector
        int32_t max_id = 0;
        for (auto& [token_str, id_val] : vocab_obj->obj) {
            int32_t id = static_cast<int32_t>(id_val.as_int());
            max_id = std::max(max_id, id);
        }

        tok_data_.vocab.resize(max_id + 1);
        for (auto& [token_str, id_val] : vocab_obj->obj) {
            int32_t id = static_cast<int32_t>(id_val.as_int());
            tok_data_.vocab[id] = token_str;
            tok_data_.token_to_id[token_str] = id;
        }
    }

    // ── Merges ───────────────────────────────────────────
    auto* merges_arr = model->get("merges");
    if (merges_arr && merges_arr->type == JsonValue::ARRAY) {
        for (auto& entry : merges_arr->arr) {
            if (entry.type == JsonValue::STRING) {
                // Format: "left right" (space-separated)
                auto& s = entry.str;
                auto sp = s.find(' ');
                if (sp != std::string::npos) {
                    tok_data_.merges.emplace_back(
                        s.substr(0, sp), s.substr(sp + 1));
                }
            } else if (entry.type == JsonValue::ARRAY && entry.arr.size() >= 2) {
                // Format: ["left", "right"]
                tok_data_.merges.emplace_back(
                    entry.arr[0].str, entry.arr[1].str);
            }
        }
    }

    // ── Added tokens (special tokens) ────────────────────
    auto* added = root.get("added_tokens");
    if (added && added->type == JsonValue::ARRAY) {
        for (auto& tok : added->arr) {
            auto* id_v      = tok.get("id");
            auto* content_v = tok.get("content");
            if (!id_v || !content_v) continue;

            int32_t id = static_cast<int32_t>(id_v->as_int());
            std::string content = content_v->as_str();

            // Extend vocab if needed
            if (id >= static_cast<int32_t>(tok_data_.vocab.size()))
                tok_data_.vocab.resize(id + 1);
            tok_data_.vocab[id] = content;
            tok_data_.token_to_id[content] = id;

            // Identify special tokens
            if (content == "<|endoftext|>")   tok_data_.bos_id = id;
            if (content == "<|im_end|>")      { tok_data_.eos_id = id; tok_data_.im_end_id = id; }
            if (content == "<|im_start|>")    tok_data_.im_start_id = id;
            if (content == "<think>")         tok_data_.think_start_id = id;
            if (content == "</think>")        tok_data_.think_end_id = id;
        }
    }

    std::cout << "[Tokenizer] vocab=" << tok_data_.vocab.size()
              << " merges=" << tok_data_.merges.size()
              << " bos=" << tok_data_.bos_id
              << " eos=" << tok_data_.eos_id << "\n";
}

// ── Tensor access ────────────────────────────────────────────

const TensorInfo* ModelLoader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

std::vector<std::string> ModelLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (auto& [k, v] : tensors_) names.push_back(k);
    std::sort(names.begin(), names.end());
    return names;
}
