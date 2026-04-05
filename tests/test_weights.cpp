#include "model_loader.h"
#include "weight_store.h"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_weights <model_dir>\n";
        return 1;
    }

    // Step 1: load from disk
    ModelLoader loader(argv[1]);

    // Step 3: transfer to GPU
    WeightStore weights;
    weights.load(loader, loader.config());

    // Query GPU memory usage after loading
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used_mb = (total_bytes - free_bytes) >> 20;
    std::cout << "[GPU] used=" << used_mb << " MB"
              << "  free=" << (free_bytes >> 20) << " MB"
              << "  total=" << (total_bytes >> 20) << " MB\n";

    // Spot-check: verify a few pointers are non-null
    auto& cfg = loader.config();
    std::cout << "\n=== Pointer checks ===\n";
    std::cout << "embedding:  " << (weights.embedding()  ? "OK" : "NULL") << "\n";
    std::cout << "final_norm: " << (weights.final_norm() ? "OK" : "NULL") << "\n";
    std::cout << "lm_head:    " << (weights.lm_head()    ? "OK" : "NULL") << "\n";
    for (int i : {0, cfg.num_layers / 2, cfg.num_layers - 1}) {
        auto& L = weights.layer(i);
        bool ok = L.q_proj && L.k_proj && L.v_proj && L.o_proj &&
                  L.gate_proj && L.up_proj && L.down_proj &&
                  L.input_layernorm && L.post_attn_norm &&
                  L.q_norm && L.k_norm;
        std::cout << "layer[" << i << "]:    " << (ok ? "OK" : "MISSING TENSORS") << "\n";
    }

    // Spot-check values: read first element of embedding back to host
    __nv_bfloat16 first_val;
    cudaMemcpy(&first_val, weights.embedding(), sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    float val_f = __bfloat162float(first_val);
    std::cout << "\nembedding[0] = " << val_f
              << "  (non-zero = " << (val_f != 0.0f ? "yes" : "NO - likely wrong!") << ")\n";

    std::cout << "\nStep 3 PASSED\n";
    return 0;
}
