#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <algorithm>

namespace gemmul8_avx512 {
namespace matmult {

inline void sign_flip_inplace(std::int8_t *A, std::size_t nbytes) {
    const __m512i mask = _mm512_set1_epi8((char)0x80);
    std::size_t   i    = 0;
    for (; i + 64 <= nbytes; i += 64) {
        __m512i v = _mm512_loadu_si512(A + i);
        _mm512_storeu_si512(A + i, _mm512_xor_si512(v, mask));
    }
    for (; i < nbytes; ++i) A[i] ^= (std::int8_t)0x80;
}

inline void pack_A(const std::int8_t *A_lo,
                   std::size_t        k_pad,
                   std::size_t        m,
                   std::size_t        m_pad,
                   std::int8_t       *A_pk,
                   bool               sign_flip = false) {
    const std::size_t m_tiles  = m_pad / 16;
    const std::size_t k_blocks = k_pad / 4;
    const std::int32_t sign_mask = sign_flip ? (std::int32_t)0x80808080u : 0;

#pragma omp parallel for schedule(static)
    for (std::size_t t = 0; t < m_tiles; ++t) {
        std::int8_t *dst_t = A_pk + t * k_pad * 16;
        for (std::size_t kb = 0; kb < k_blocks; ++kb) {
            std::int8_t *dst = dst_t + kb * 64;
            for (std::size_t lane = 0; lane < 16; ++lane) {
                std::size_t m_idx = t * 16 + lane;
                std::int32_t v     = 0;
                if (m_idx < m) {
                    std::memcpy(&v, A_lo + m_idx * k_pad + kb * 4, 4);
                    v ^= sign_mask;
                }
                std::memcpy(dst + lane * 4, &v, 4);
            }
        }
    }
}

inline void compute_S_B(const std::int8_t *B_lo,
                        std::size_t        k_pad,
                        std::size_t        n,
                        std::size_t        n_pad,
                        std::int32_t      *S_B) {
#pragma omp parallel for schedule(static)
    for (std::size_t nn = 0; nn < n_pad; ++nn) {
        if (nn >= n) { S_B[nn] = 0; continue; }
        const std::int8_t *col = B_lo + nn * k_pad;
        __m512i            acc = _mm512_setzero_si512();
        std::size_t        i   = 0;
        const __m512i ones = _mm512_set1_epi8(1);
        for (; i + 64 <= k_pad; i += 64) {
            __m512i v = _mm512_load_si512(col + i);
            acc       = _mm512_dpbusd_epi32(acc, ones, v);
        }
        int s = _mm512_reduce_add_epi32(acc);
        for (; i < k_pad; ++i) s += col[i];
        S_B[nn] = s;
    }
}

inline void kernel_16x8(const std::int8_t  *A_pk_tile,
                        const std::int8_t  *B_lo,
                        std::size_t         k_pad,
                        std::int32_t       *C,
                        std::size_t         ldc,
                        const std::int32_t *S_B_neg128) {
    __m512i c0 = _mm512_setzero_si512();
    __m512i c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512();
    __m512i c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512();
    __m512i c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512();
    __m512i c7 = _mm512_setzero_si512();

    const std::int8_t *b0 = B_lo;
    const std::int8_t *b1 = B_lo + 1 * k_pad;
    const std::int8_t *b2 = B_lo + 2 * k_pad;
    const std::int8_t *b3 = B_lo + 3 * k_pad;
    const std::int8_t *b4 = B_lo + 4 * k_pad;
    const std::int8_t *b5 = B_lo + 5 * k_pad;
    const std::int8_t *b6 = B_lo + 6 * k_pad;
    const std::int8_t *b7 = B_lo + 7 * k_pad;

    for (std::size_t k = 0; k < k_pad; k += 4) {
        __m512i av = _mm512_load_si512(A_pk_tile + k * 16);

        __m512i bv0 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b0 + k));
        __m512i bv1 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b1 + k));
        __m512i bv2 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b2 + k));
        __m512i bv3 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b3 + k));
        __m512i bv4 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b4 + k));
        __m512i bv5 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b5 + k));
        __m512i bv6 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b6 + k));
        __m512i bv7 = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(b7 + k));

        c0 = _mm512_dpbusd_epi32(c0, av, bv0);
        c1 = _mm512_dpbusd_epi32(c1, av, bv1);
        c2 = _mm512_dpbusd_epi32(c2, av, bv2);
        c3 = _mm512_dpbusd_epi32(c3, av, bv3);
        c4 = _mm512_dpbusd_epi32(c4, av, bv4);
        c5 = _mm512_dpbusd_epi32(c5, av, bv5);
        c6 = _mm512_dpbusd_epi32(c6, av, bv6);
        c7 = _mm512_dpbusd_epi32(c7, av, bv7);
    }

    _mm512_store_si512(C + 0 * ldc, _mm512_add_epi32(c0, _mm512_set1_epi32(S_B_neg128[0])));
    _mm512_store_si512(C + 1 * ldc, _mm512_add_epi32(c1, _mm512_set1_epi32(S_B_neg128[1])));
    _mm512_store_si512(C + 2 * ldc, _mm512_add_epi32(c2, _mm512_set1_epi32(S_B_neg128[2])));
    _mm512_store_si512(C + 3 * ldc, _mm512_add_epi32(c3, _mm512_set1_epi32(S_B_neg128[3])));
    _mm512_store_si512(C + 4 * ldc, _mm512_add_epi32(c4, _mm512_set1_epi32(S_B_neg128[4])));
    _mm512_store_si512(C + 5 * ldc, _mm512_add_epi32(c5, _mm512_set1_epi32(S_B_neg128[5])));
    _mm512_store_si512(C + 6 * ldc, _mm512_add_epi32(c6, _mm512_set1_epi32(S_B_neg128[6])));
    _mm512_store_si512(C + 7 * ldc, _mm512_add_epi32(c7, _mm512_set1_epi32(S_B_neg128[7])));
}

inline void kernel_16xNr(const std::int8_t  *A_pk_tile,
                         const std::int8_t  *B_lo,
                         std::size_t         k_pad,
                         std::int32_t       *C,
                         std::size_t         ldc,
                         const std::int32_t *S_B_neg128,
                         int                 nr) {
    __m512i acc[8];
    for (int j = 0; j < nr; ++j) acc[j] = _mm512_setzero_si512();

    for (std::size_t k = 0; k < k_pad; k += 4) {
        __m512i av = _mm512_loadu_si512(A_pk_tile + k * 16);
        for (int j = 0; j < nr; ++j) {
            __m512i bv = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(B_lo + j * k_pad + k));
            acc[j]     = _mm512_dpbusd_epi32(acc[j], av, bv);
        }
    }
    for (int j = 0; j < nr; ++j) {
        _mm512_storeu_si512(C + j * ldc,
                            _mm512_add_epi32(acc[j], _mm512_set1_epi32(S_B_neg128[j])));
    }
}

inline void kernel_edge(const std::int8_t  *A_pk_tile,
                        const std::int8_t  *B_lo,
                        std::size_t         k_pad,
                        std::int32_t       *C,
                        std::size_t         ldc,
                        const std::int32_t *S_B_neg128,
                        int                 m_r,
                        int                 n_r) {
    __m512i acc[8];
    for (int j = 0; j < n_r; ++j) acc[j] = _mm512_setzero_si512();

    for (std::size_t k = 0; k < k_pad; k += 4) {
        __m512i av = _mm512_loadu_si512(A_pk_tile + k * 16);
        for (int j = 0; j < n_r; ++j) {
            __m512i bv = _mm512_set1_epi32(*reinterpret_cast<const std::int32_t *>(B_lo + j * k_pad + k));
            acc[j]     = _mm512_dpbusd_epi32(acc[j], av, bv);
        }
    }
    __mmask16 mmask = (m_r >= 16) ? 0xFFFFu : (__mmask16)((1u << m_r) - 1);
    for (int j = 0; j < n_r; ++j) {
        __m512i v = _mm512_add_epi32(acc[j], _mm512_set1_epi32(S_B_neg128[j]));
        _mm512_mask_storeu_epi32(C + j * ldc, mmask, v);
    }
}

inline void gemm_i8x1(const std::int8_t  *A_pk,
                      const std::int8_t  *B_lo,
                      std::int32_t       *C,
                      std::size_t         ldc,
                      std::size_t         m,
                      std::size_t         n,
                      std::size_t         k_pad,
                      std::size_t         m_pad,
                      std::size_t         n_pad,
                      const std::int32_t *S_B_neg128) {
    (void)n_pad;
    const std::size_t m_tiles = m_pad / 16;

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t n0 = 0; n0 < n; n0 += 8) {
        for (std::size_t t = 0; t < m_tiles; ++t) {
            std::size_t m0 = t * 16;
            if (m0 >= m) continue;
            int m_r = (int)std::min<std::size_t>(16, m - m0);
            int n_r = (int)std::min<std::size_t>(8, n - n0);
            if (n_r <= 0) continue;
            const std::int8_t  *A_b  = A_pk + t * k_pad * 16;
            const std::int8_t  *B_b  = B_lo + n0 * k_pad;
            std::int32_t       *C_b  = C + n0 * ldc + m0;
            const std::int32_t *S_Bb = S_B_neg128 + n0;
            if (m_r == 16 && n_r == 8) {
                kernel_16x8(A_b, B_b, k_pad, C_b, ldc, S_Bb);
            } else if (m_r == 16) {
                kernel_16xNr(A_b, B_b, k_pad, C_b, ldc, S_Bb, n_r);
            } else {
                kernel_edge(A_b, B_b, k_pad, C_b, ldc, S_Bb, m_r, n_r);
            }
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
            std::int32_t a = std::abs(col[i]);
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

} // namespace matmult
} // namespace gemmul8_avx512
