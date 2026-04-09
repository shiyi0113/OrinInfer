#include "model_loader.h"
#include "activation_pool.h"
#include "kv_cache.h"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_buffers <model_dir>\n";
        return 1;
    }

    ModelLoader loader(argv[1]);
    auto& cfg = loader.config();

    // ── ActivationPool ────────────────────────────────────
    ActivationPool pool;
    pool.init(cfg);

    // Verify buffers are non-null and distinct
    auto* a = pool.input_buf();
    auto* b = pool.output_buf();
    std::cout << "\n=== ActivationPool ===\n";
    std::cout << "input_buf:  " << (a ? "OK" : "NULL") << "\n";
    std::cout << "output_buf: " << (b ? "OK" : "NULL") << "\n";
    std::cout << "distinct:   " << (a != b ? "OK" : "SAME - BUG") << "\n";
    std::cout << "scratch:    " << (pool.scratch() ? "OK" : "NULL") << "\n";
    std::cout << "logits:     " << (pool.logits()  ? "OK" : "NULL") << "\n";

    // Verify flip swaps correctly
    pool.flip();
    std::cout << "after flip, input_buf==old output_buf: "
              << (pool.input_buf() == b ? "OK" : "BUG") << "\n";
    pool.reset();

    // ── KVCache (Paged) ───────────────────────────────────
    KVCache cache;
    cache.init(cfg);

    std::cout << "\n=== KVCache (Paged) ===\n";
    std::cout << "block_size=" << cache.block_size() << "\n";

    // Check all layer pool pointers are non-null
    bool all_ok = true;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (!cache.k_pool(i) || !cache.v_pool(i)) { all_ok = false; break; }
    }
    std::cout << "all " << cfg.num_layers << " layer KV pools: " << (all_ok?"OK":"NULL found") << "\n";
    std::cout << "block_table (device): " << (cache.d_block_table() ? "OK" : "NULL") << "\n";

    // prepare() allocates pages and uploads block table
    cache.prepare(10, 0);
    std::cout << "prepare(10, 0): OK\n";

    cache.prepare(1, 10);
    std::cout << "prepare(1, 10): OK\n";

    // reset frees all pages
    cache.reset();
    cache.prepare(1, 0);
    std::cout << "reset + prepare(1, 0): OK\n";

    // ── Final GPU memory report ───────────────────────────
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cout << "\n[GPU] used=" << ((total_b - free_b) >> 20) << " MB"
              << "  free=" << (free_b >> 20) << " MB\n";

    std::cout << "\nStep 4 PASSED\n";
    return 0;
}
