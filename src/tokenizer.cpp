#include "tokenizer.h"
#include <iostream>
#include <cstring>
#include <cctype>

// ═══════════════════════════════════════════════════════════════
// GPT-2 byte ↔ Unicode tables and pre-tokenizer
// ═══════════════════════════════════════════════════════════════
namespace {

// byte_to_unicode[b] = Unicode codepoint for byte b
// Computed once from GPT-2 formula:
//   Printable ranges (33-126, 161-172, 174-255) map to themselves.
//   The remaining 68 bytes (0-32, 127-160, 173) map to U+0100..U+0143.
static uint32_t g_b2u[256];
static uint32_t g_u2b[0x144]; // inverse: indexed by codepoint (fits in 0x144 entries)
static bool     g_tables_ready = false;

static void init_tables() {
    if (g_tables_ready) return;

    bool is_direct[256] = {};
    for (int b = 33;   b <= 126; b++) is_direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) is_direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) is_direct[b] = true;

    uint32_t next_cp = 256; // 0x100
    for (int b = 0; b < 256; b++) {
        uint32_t cp = is_direct[b] ? static_cast<uint32_t>(b) : next_cp++;
        g_b2u[b] = cp;
        // inverse table: all codepoints fall in range [33, 0x143]
        if (cp < 0x144) g_u2b[cp] = static_cast<uint32_t>(b);
    }

    g_tables_ready = true;
}

// ── UTF-8 helpers ─────────────────────────────────────────────

// Read one Unicode codepoint from UTF-8 at *p, advance p.
static uint32_t utf8_read(const char*& p) {
    unsigned char c = static_cast<unsigned char>(*p++);
    if (c < 0x80) return c;
    if (c < 0xE0) {
        uint32_t cp = (c & 0x1F) << 6;
        cp |= static_cast<unsigned char>(*p++) & 0x3F;
        return cp;
    }
    if (c < 0xF0) {
        uint32_t cp = (c & 0x0F) << 12;
        cp |= (static_cast<unsigned char>(*p++) & 0x3F) << 6;
        cp |= static_cast<unsigned char>(*p++) & 0x3F;
        return cp;
    }
    uint32_t cp = (c & 0x07) << 18;
    cp |= (static_cast<unsigned char>(*p++) & 0x3F) << 12;
    cp |= (static_cast<unsigned char>(*p++) & 0x3F) << 6;
    cp |= static_cast<unsigned char>(*p++) & 0x3F;
    return cp;
}

// Encode one Unicode codepoint to UTF-8.
static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// ── Unicode category approximations ──────────────────────────
// Used to implement the Qwen2 regex split pattern without <regex>.
// \p{L}: Unicode letters (covers CJK, Latin, Cyrillic, Arabic, etc.)
// \p{N}: Unicode digits (ASCII 0-9 only needed in practice)

static bool is_letter(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp < 0x80) return false;
    if (cp >= 0x00C0 && cp <= 0x02FF) return true; // Latin Extended + IPA
    if (cp >= 0x0370 && cp <= 0x04FF) return true; // Greek, Cyrillic
    if (cp >= 0x0500 && cp <= 0x052F) return true; // Cyrillic Supplement
    if (cp >= 0x0530 && cp <= 0x058F) return true; // Armenian
    if (cp >= 0x0590 && cp <= 0x06FF) return true; // Hebrew, Arabic
    if (cp >= 0x0900 && cp <= 0x0DFF) return true; // Indic scripts
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true; // Thai
    if (cp >= 0x3040 && cp <= 0x30FF) return true; // Hiragana/Katakana
    if (cp >= 0x3400 && cp <= 0x9FFF) return true; // CJK Unified + Ext A
    if (cp >= 0xA000 && cp <= 0xA4CF) return true; // Yi
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true; // Hangul Syllables
    if (cp >= 0xF900 && cp <= 0xFAFF) return true; // CJK Compatibility
    if (cp >= 0x20000 && cp <= 0x2FA1F) return true; // CJK Ext B-F
    return false;
}

static bool is_digit(uint32_t cp)  { return cp >= '0' && cp <= '9'; }
static bool is_space(uint32_t cp)  { return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'; }

// ── Pre-tokenizer ─────────────────────────────────────────────
// Implements the Qwen2 tokenizer split pattern (Sequence of Split+ByteLevel):
//   Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)       -- contractions
//   Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+            -- word (opt leading non-word char)
//   Rule 3: \p{N}                                -- single digit
//   Rule 4: [ ]?[^\s\p{L}\p{N}]+[\r\n]*         -- punctuation / symbols
//   Rule 5: \s*[\r\n]+                           -- newlines
//   Rule 6: \s+                                  -- remaining whitespace

static size_t match_contraction(const char* p) {
    if (*p != '\'') return 0;
    const char* q = p + 1;
    if (!*q) return 0;
    char a = static_cast<char>(tolower(static_cast<unsigned char>(*q)));
    // Two-character: 're, 've, 'll
    if (*(q+1)) {
        char b = static_cast<char>(tolower(static_cast<unsigned char>(*(q+1))));
        if ((a=='r'&&b=='e') || (a=='v'&&b=='e') || (a=='l'&&b=='l')) return 3;
    }
    // One-character: 's, 't, 'm, 'd
    if (a=='s' || a=='t' || a=='m' || a=='d') return 2;
    return 0;
}

static std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> segs;
    const char* p   = text.c_str();
    const char* end = p + text.size();

    while (p < end) {
        // ── Rule 1: contractions ──────────────────────────
        {
            size_t n = match_contraction(p);
            if (n) { segs.emplace_back(p, n); p += n; continue; }
        }

        // Decode current codepoint (without advancing p yet)
        const char* p1 = p;
        uint32_t    cp = utf8_read(p1); // p1 is now past cp

        // ── Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+ ────────────
        {
            const char* q = p;
            bool has_leading = false;

            // Optional one non-letter non-digit non-newline leading char
            if (!is_letter(cp) && !is_digit(cp) && cp != '\r' && cp != '\n') {
                if (p1 < end) {
                    const char* p2 = p1;
                    uint32_t c1 = utf8_read(p2);
                    if (is_letter(c1)) {
                        q = p1;       // skip past leading char
                        has_leading = true;
                    }
                }
            }

            // Consume consecutive letters
            const char* letter_start = q;
            while (q < end) {
                const char* r = q;
                uint32_t c = utf8_read(r);
                if (is_letter(c)) q = r; else break;
            }

            if (q > letter_start) {
                segs.emplace_back(p, q - p);
                p = q;
                continue;
            }
            // has_leading but no letters: fall through (shouldn't happen by construction)
        }

        // ── Rule 3: single digit ──────────────────────────
        if (is_digit(cp)) {
            segs.emplace_back(p, p1 - p);
            p = p1;
            continue;
        }

        // ── Rule 4: [ ]?[^\s\p{L}\p{N}]+[\r\n]* ─────────
        {
            const char* q = p;
            // Optional leading space (only if followed by punct)
            if (cp == ' ' && p1 < end) {
                const char* p2 = p1;
                uint32_t c1 = utf8_read(p2);
                if (!is_space(c1) && !is_letter(c1) && !is_digit(c1))
                    q = p1; // include the space
            }

            const char* punct_start = q;
            while (q < end) {
                const char* r = q;
                uint32_t c = utf8_read(r);
                if (!is_space(c) && !is_letter(c) && !is_digit(c)) q = r;
                else break;
            }

            if (q > punct_start) {
                // Optional trailing [\r\n]*
                while (q < end) {
                    const char* r = q;
                    uint32_t c = utf8_read(r);
                    if (c == '\r' || c == '\n') q = r; else break;
                }
                segs.emplace_back(p, q - p);
                p = q;
                continue;
            }
        }

        // ── Rule 5/6: whitespace (including newlines) ─────
        {
            const char* q = p1; // already past cp
            while (q < end) {
                const char* r = q;
                uint32_t c = utf8_read(r);
                if (is_space(c)) q = r; else break;
            }
            segs.emplace_back(p, q - p);
            p = q;
        }
    }

    return segs;
}

} // namespace

// ═══════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════

Tokenizer::Tokenizer(const TokenizerData& data) {
    init_tables();

    id_to_token_ = data.vocab;
    token_to_id_ = data.token_to_id;

    bos_id_      = data.bos_id;
    eos_id_      = data.eos_id;
    im_start_id_ = data.im_start_id;
    im_end_id_   = data.im_end_id;

    // Build merge table: (left_id, right_id) → (merged_id, priority)
    int built = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(data.merges.size()); i++) {
        auto& [ls, rs] = data.merges[i];
        auto li = token_to_id_.find(ls);
        auto ri = token_to_id_.find(rs);
        auto mi = token_to_id_.find(ls + rs);
        if (li == token_to_id_.end() || ri == token_to_id_.end() ||
            mi == token_to_id_.end()) continue;
        merges_[{li->second, ri->second}] = {mi->second, i};
        built++;
    }

    std::cout << "[Tokenizer] vocab=" << id_to_token_.size()
              << " merges_built=" << built << "\n";
}

// ── encode ─────────────────────────────────────────────────────
// 1. Pre-tokenize text into segments (matching Qwen2 split pattern)
// 2. Per segment: bytes → byte-level Unicode token ids
// 3. Apply BPE merges within each segment
std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> result;

    for (auto& seg : pretokenize(text)) {
        // Convert each byte to its byte-level Unicode token id
        std::vector<int32_t> ids;
        ids.reserve(seg.size());

        for (unsigned char b : seg) {
            uint32_t cp = g_b2u[b];
            std::string tok = cp_to_utf8(cp);
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
            // Unknown byte: silently skip (should not happen)
        }

        bpe_encode(ids);
        result.insert(result.end(), ids.begin(), ids.end());
    }

    return result;
}

// ── bpe_encode ─────────────────────────────────────────────────
// Iteratively merge the highest-priority adjacent pair.
void Tokenizer::bpe_encode(std::vector<int32_t>& tokens) const {
    while (tokens.size() >= 2) {
        int best_idx      = -1;
        int best_priority = INT32_MAX;

        for (int i = 0; i < static_cast<int>(tokens.size()) - 1; i++) {
            auto it = merges_.find({tokens[i], tokens[i + 1]});
            if (it != merges_.end() && it->second.priority < best_priority) {
                best_priority = it->second.priority;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;

        auto it = merges_.find({tokens[best_idx], tokens[best_idx + 1]});
        tokens[best_idx] = it->second.merged_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
}

// ── decode ─────────────────────────────────────────────────────
// Token ids → concatenate byte-level Unicode strings → bytes → UTF-8
std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Concatenate raw token strings (GPT-2 byte-level Unicode)
    std::string raw;
    for (auto id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) continue;
        if (id >= 151643) continue;  // special token: skip in output
        raw += id_to_token_[id];
    }

    // Decode byte-level Unicode codepoints back to raw bytes
    std::string out;
    out.reserve(raw.size());
    const char* p   = raw.c_str();
    const char* end = p + raw.size();
    while (p < end) {
        uint32_t cp = utf8_read(p);
        if (cp < 0x144)
            out += static_cast<char>(g_u2b[cp]);
        // Codepoints outside the byte-map range: skip (should not occur)
    }
    return out;
}

// ── decode_token ───────────────────────────────────────────────
// Single token id → decoded string piece (for streaming output).
// Returns "" for special tokens.
std::string Tokenizer::decode_token(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) return "";
    if (id >= 151643) return "";  // special token

    const auto& tok = id_to_token_[id];
    std::string out;
    const char* p   = tok.c_str();
    const char* end = p + tok.size();
    while (p < end) {
        uint32_t cp = utf8_read(p);
        if (cp < 0x144)
            out += static_cast<char>(g_u2b[cp]);
    }
    return out;
}

// ── apply_chat_template ────────────────────────────────────────
// Qwen3 chat format (non-thinking mode):
//   <|im_start|>system\n{sys}<|im_end|>\n
//   <|im_start|>user\n{message}<|im_end|>\n
//   <|im_start|>assistant\n<think>\n\n</think>\n
//
// The empty <think>\n\n</think> block tells Qwen3 to skip chain-of-thought
// and produce a direct answer. Without it the model enters thinking mode
// and uses the entire context budget on internal reasoning.
std::vector<int32_t> Tokenizer::apply_chat_template(const std::string& user_msg,
                                                      bool enable_thinking) const {
    std::vector<int32_t> ids;

    auto push = [&](int32_t id) { ids.push_back(id); };
    auto push_str = [&](const std::string& s) {
        auto toks = encode(s);
        ids.insert(ids.end(), toks.begin(), toks.end());
    };

    // System turn
    push(im_start_id_);
    push_str("system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
    push(im_end_id_);
    push_str("\n");

    // User turn
    push(im_start_id_);
    push_str("user\n" + user_msg);
    push(im_end_id_);
    push_str("\n");

    // Assistant turn prefix
    push(im_start_id_);
    push_str("assistant\n");

    // Suppress chain-of-thought by prefilling an empty think block
    if (!enable_thinking) {
        push_str("<think>\n\n</think>\n");
    }

    return ids;
}
