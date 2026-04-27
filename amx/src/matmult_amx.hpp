#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace gemmul8_amx {
namespace matmult_amx {

inline void scalar_tile(const std::int8_t *A_lo,
                        const std::int8_t *B_lo,
                        std::int32_t      *C,
                        std::size_t        ldc,
                        std::size_t        k_pad,
                        int                m_r,
                        int                n_r) {
    for (int j = 0; j < n_r; ++j) {
        for (int i = 0; i < m_r; ++i) {
            std::int32_t acc = 0;
            for (std::size_t k = 0; k < k_pad; ++k) {
                acc += (std::int32_t)A_lo[i * k_pad + k] *
                       (std::int32_t)B_lo[j * k_pad + k];
            }
            C[j * ldc + i] = acc;
        }
    }
}

inline void c_hi_col_absmax(const std::int32_t *C_hi,
                            std::size_t         m,
                            std::size_t         n,
                            std::size_t         m_pad,
                            std::int32_t       *out_per_col) {
#pragma omp parallel for schedule(static)
    for (std::size_t c = 0; c < n; ++c) {
        const std::int32_t *col = C_hi + c * m_pad;
        __m512i             vm  = _mm512_setzero_si512();
        std::size_t         i   = 0;
        for (; i + 16 <= m; i += 16) {
            __m512i v = _mm512_loadu_si512(col + i);
            __m512i a = _mm512_abs_epi32(v);
            vm        = _mm512_max_epi32(vm, a);
        }
        std::int32_t mx = _mm512_reduce_max_epi32(vm);
        for (; i < m; ++i) {
            std::int32_t x = col[i];
            std::int32_t a = (x < 0) ? -x : x;
            if (a > mx) mx = a;
        }
        out_per_col[c] = mx;
    }
}

inline void c_hi_row_absmax(const std::int32_t *C_hi,
                            std::size_t         m,
                            std::size_t         n,
                            std::size_t         m_pad,
                            std::int32_t       *out_per_row) {
#pragma omp parallel for schedule(static)
    for (std::size_t r0 = 0; r0 < m; r0 += 16) {
        std::size_t r_end = std::min<std::size_t>(r0 + 16, m);
        int         lanes = (int)(r_end - r0);
        __mmask16   mm    = (lanes >= 16) ? 0xFFFFu : (__mmask16)((1u << lanes) - 1);
        __m512i     vm    = _mm512_setzero_si512();
        for (std::size_t c = 0; c < n; ++c) {
            __m512i v = _mm512_maskz_loadu_epi32(mm, C_hi + c * m_pad + r0);
            __m512i a = _mm512_abs_epi32(v);
            vm        = _mm512_max_epi32(vm, a);
        }
        alignas(64) std::int32_t tmp[16];
        _mm512_storeu_si512(tmp, vm);
        for (int i = 0; i < lanes; ++i) out_per_row[r0 + i] = tmp[i];
    }
}

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)

struct TileConfig {
    std::uint8_t  palette_id;
    std::uint8_t  start_row;
    std::uint8_t  reserved[14];
    std::uint16_t colsb[16];
    std::uint8_t  rows[16];
};

inline void load_config_16x16x64() {
    alignas(64) TileConfig cfg{};
    cfg.palette_id = 1;

    cfg.rows[0]  = 16;
    cfg.colsb[0] = 64;
    cfg.rows[3]  = 16;
    cfg.colsb[3] = 64;
    cfg.rows[4]  = 16;
    cfg.colsb[4] = 64;
    cfg.rows[5]  = 16;
    cfg.colsb[5] = 64;

    cfg.rows[1]  = 16;
    cfg.colsb[1] = 64;

    cfg.rows[2]  = 16;
    cfg.colsb[2] = 64;

    _tile_loadconfig(&cfg);
}

inline void pack_B_16x64(const std::int8_t *B_lo,
                         std::size_t       k_pad,
                         std::size_t       k0,
                         std::int8_t      *B_tile) {
    for (int kb4 = 0; kb4 < 16; ++kb4) {
        std::int8_t *dst = B_tile + kb4 * 64;
        const std::size_t kk = k0 + (std::size_t)kb4 * 4;
        for (int j = 0; j < 16; ++j) {
            std::memcpy(dst + j * 4, B_lo + (std::size_t)j * k_pad + kk, 4);
        }
    }
}

inline void kernel_16x16(const std::int8_t *A_lo_tile,
                         const std::int8_t *B_lo_tile,
                         std::size_t        k_pad,
                         std::int32_t      *C,
                         std::size_t        ldc) {
    alignas(64) std::int8_t  B_tile[16 * 64];
    alignas(64) std::int32_t C_tile[16 * 16];

    _tile_zero(0);
    for (std::size_t k0 = 0; k0 < k_pad; k0 += 64) {
        pack_B_16x64(B_lo_tile, k_pad, k0, B_tile);
        _tile_loadd(1, A_lo_tile + k0, k_pad);
        _tile_loadd(2, B_tile, 64);
        _tile_dpbssd(0, 1, 2);
    }
    _tile_stored(0, C_tile, 64);

    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            C[j * ldc + i] = C_tile[i * 16 + j];
        }
    }
}

inline void store_tile_colmajor(const std::int32_t *C_tile,
                                std::int32_t       *C,
                                std::size_t         ldc) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            C[j * ldc + i] = C_tile[i * 16 + j];
        }
    }
}

inline void kernel_64x16(const std::int8_t *A_lo_tile,
                         const std::int8_t *B_lo_tile,
                         std::size_t        k_pad,
                         std::int32_t      *C,
                         std::size_t        ldc) {
    alignas(64) std::int8_t  B_tile[16 * 64];
    alignas(64) std::int32_t C_tile0[16 * 16];
    alignas(64) std::int32_t C_tile1[16 * 16];
    alignas(64) std::int32_t C_tile2[16 * 16];
    alignas(64) std::int32_t C_tile3[16 * 16];

    const std::int8_t *A0 = A_lo_tile;
    const std::int8_t *A1 = A0 + 16 * k_pad;
    const std::int8_t *A2 = A1 + 16 * k_pad;
    const std::int8_t *A3 = A2 + 16 * k_pad;

    _tile_zero(0);
    _tile_zero(3);
    _tile_zero(4);
    _tile_zero(5);

    for (std::size_t k0 = 0; k0 < k_pad; k0 += 64) {
        pack_B_16x64(B_lo_tile, k_pad, k0, B_tile);
        _tile_loadd(2, B_tile, 64);

        _tile_loadd(1, A0 + k0, k_pad);
        _tile_dpbssd(0, 1, 2);
        _tile_loadd(1, A1 + k0, k_pad);
        _tile_dpbssd(3, 1, 2);
        _tile_loadd(1, A2 + k0, k_pad);
        _tile_dpbssd(4, 1, 2);
        _tile_loadd(1, A3 + k0, k_pad);
        _tile_dpbssd(5, 1, 2);
    }

    _tile_stored(0, C_tile0, 64);
    _tile_stored(3, C_tile1, 64);
    _tile_stored(4, C_tile2, 64);
    _tile_stored(5, C_tile3, 64);

    store_tile_colmajor(C_tile0, C, ldc);
    store_tile_colmajor(C_tile1, C + 16, ldc);
    store_tile_colmajor(C_tile2, C + 32, ldc);
    store_tile_colmajor(C_tile3, C + 48, ldc);
}

inline void gemm_i8x1(const std::int8_t *A_lo,
                      const std::int8_t *B_lo,
                      std::int32_t      *C,
                      std::size_t        ldc,
                      std::size_t        m,
                      std::size_t        n,
                      std::size_t        k_pad,
                      std::size_t        m_pad,
                      std::size_t        n_pad) {
    (void)m_pad;
    (void)n_pad;

#pragma omp parallel
    {
        load_config_16x16x64();

#pragma omp for collapse(2) schedule(static)
        for (std::size_t n0 = 0; n0 < n; n0 += 16) {
            for (std::size_t m0 = 0; m0 < m; m0 += 64) {
                int n_r = (int)std::min<std::size_t>(16, n - n0);
                const std::int8_t *A_b = A_lo + m0 * k_pad;
                const std::int8_t *B_b = B_lo + n0 * k_pad;
                std::int32_t      *C_b = C + n0 * ldc + m0;

                if (m0 + 64 <= m && n_r == 16) {
                    kernel_64x16(A_b, B_b, k_pad, C_b, ldc);
                } else {
                    std::size_t m1 = m0;
                    while (m1 < m && m1 < m0 + 64) {
                        int mt = (int)std::min<std::size_t>(16, m - m1);
                        const std::int8_t *A_t = A_lo + m1 * k_pad;
                        std::int32_t      *C_t = C + n0 * ldc + m1;
                        if (mt == 16 && n_r == 16) {
                            kernel_16x16(A_t, B_b, k_pad, C_t, ldc);
                        } else {
                            scalar_tile(A_t, B_b, C_t, ldc, k_pad, mt, n_r);
                        }
                        m1 += 16;
                    }
                }
            }
        }

        _tile_release();
    }
}

#else

inline void gemm_i8x1(const std::int8_t *,
                      const std::int8_t *,
                      std::int32_t *,
                      std::size_t,
                      std::size_t,
                      std::size_t,
                      std::size_t,
                      std::size_t,
                      std::size_t) {
}

#endif

} // namespace matmult_amx
} // namespace gemmul8_amx
