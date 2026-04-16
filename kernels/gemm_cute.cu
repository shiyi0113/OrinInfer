#include <cuda.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>

template <typename Config>
__global__ void gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n, int k)
{
    using namespace cute;

    using T = typename Config::T;
    using TiledMMA = typename Config::MMA;

    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    // Global memory tensor
    Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, Int<1>{})); // (M, K)
    Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), make_stride(k, Int<1>{})); // (N, K)
    Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n), make_stride(n, Int<1>{})); // (M, N)

    // slice
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileN, kTileK, k)
    Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix)); // (kTileM, kTileN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (kTileN, kTileK, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition method
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)

    // fill zero for accumulator
    clear(tCrD);

    // gmem -cp.async-> shm -ldmatrix-> reg
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // ? (CPY, CPY_M, CPY_K)

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    // submit kStage - 1 tile
    // gmem -> shm
#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage)
    {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    // wait one submitted gmem->smem done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
    int ntile = k / kTileK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile)
    {
        int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik)
        {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1)
            {
                cp_async_wait<kStage - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % kStage;
            }

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0)
            {
                if (itile_to_read < ntile)
                {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // for ik
    } // itile

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s); // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
    {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j)
        {
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);
            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

#pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j)
        {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }
}

namespace config
{
    using namespace cute;
    template <typename _T,
              int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
              int kStage_ = 5, int kSmemLayoutCBatch_ = 2>
    struct GemmConfig
    {
        using T = _T;
        // Tile
        static constexpr int kTileM = kTileM_;
        static constexpr int kTileN = kTileN_;
        static constexpr int kTileK = kTileK_;

        // Tile MMA
        using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;
        using mma_traits = MMA_Traits<mma_op>;
        using mma_atom   = MMA_Atom<mma_traits>;   // fix: traits, not op

        static constexpr int kMmaEURepeatM = 4;    // fix: 4 warps in M
        static constexpr int kMmaEURepeatN = 2;
        static constexpr int kMmaEURepeatK = 1;
        using MMA_EU_RepeatT = decltype(make_layout(make_shape(
            Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));

        // Partition tile: atom shape × EU repeats (× value repeats for M and N)
        using mma_atom_shape = mma_traits::Shape_MNK;     // (16, 8, 16)
        static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});  // 64
        static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});  // 32
        static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});  // 16
        using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

        using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

        // Tile Copy G2S
        using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

        using G2SCopyA = decltype(make_tiled_copy(
            g2s_copy_atom{},
            make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
            make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using G2SCopyB = G2SCopyA;

        // Tile Copy S2R
        using s2r_copy_op = SM75_U32x4_LDSM_N;
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using S2RCopyAtomA = Copy_Atom<s2r_copy_traits, T>;

        using S2RCopyAtomB = S2RCopyAtomA;

        // Tile Copy R2S
        using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
        // Tile Copy S2G
        using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
        using S2GCopyC = decltype(make_tiled_copy(
            S2GCopyAtomC{},
            make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
            make_layout(make_shape(Int<1>{}, Int<8>{}))));

        // Pipeline
        static constexpr int kStage = kStage_;
        static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;
        // Swizzle
        using SmemLayoutAtom = decltype(composition(
            Swizzle<3, 3, 3>{},
            make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                        make_stride(Int<kTileK>{}, Int<1>{}))));

        using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                                   make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));

        using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                                   make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

        using SmemLayoutAtomC = decltype(composition(
            Swizzle<2, 3, 3>{},
            make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                        make_stride(Int<kMmaPN>{}, Int<1>{}))));
        using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},
                                                   make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

        static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                      "C shared memory request is larger than A's one pipe");

        // 其他参数
        static constexpr int kThreadNum = size(MMA{});   // 256 (4×2 warps × 32 threads)
        static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
        static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
        static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
    };
}

// ── Host-side launcher ────────────────────────────────────────────────────────
//
// Default config: kTileM=128, kTileN=128, kTileK=32, kStage=5
// kThreadNum=256 (kMmaEURepeatM=4, kMmaEURepeatN=2 → 8 warps × 32 threads)
//
// Returns true if the kernel was launched (M/N/K tile-aligned),
// false if dimensions are not aligned — caller should use a fallback.
bool launch_gemm_cute(__nv_bfloat16* C,
                      const __nv_bfloat16* A,
                      const __nv_bfloat16* B,
                      int M, int N, int K) {
    using namespace cute;
    using Cfg = config::GemmConfig<bfloat16_t>;   // default tile 128×128×32, 5 stages

    if (M % Cfg::kTileM != 0 || N % Cfg::kTileN != 0 || K % Cfg::kTileK != 0)
        return false;

    static bool smem_set = false;
    if (!smem_set) {
        cudaFuncSetAttribute(gemm_multi_stage<Cfg>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Cfg::kShmSize);
        smem_set = true;
    }

    dim3 block(Cfg::kThreadNum);
    dim3 grid((N + Cfg::kTileN - 1) / Cfg::kTileN,
              (M + Cfg::kTileM - 1) / Cfg::kTileM);

    gemm_multi_stage<Cfg><<<grid, block, Cfg::kShmSize>>>(
        static_cast<void*>(C),
        static_cast<const void*>(A),
        static_cast<const void*>(B),
        M, N, K);

    return true;
}
