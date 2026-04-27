#pragma once
#include "common.hpp"
#include "math_helpers.hpp"
#include "table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace gemmul8_amx {
namespace scaling {

inline __m512d load_pd(const double *p) { return _mm512_loadu_pd(p); }

inline __m512i bcast_i32(int32_t x) { return _mm512_set1_epi32(x); }

inline double hmax_pd(__m512d v) {
    __m256d lo  = _mm512_castpd512_pd256(v);
    __m256d hi  = _mm512_extractf64x4_pd(v, 1);
    __m256d max = _mm256_max_pd(lo, hi);
    __m128d a   = _mm256_castpd256_pd128(max);
    __m128d b   = _mm256_extractf128_pd(max, 1);
    __m128d m   = _mm_max_pd(a, b);
    __m128d sh  = _mm_unpackhi_pd(m, m);
    return _mm_cvtsd_f64(_mm_max_sd(m, sh));
}

inline double hsum_pd(__m512d v) { return _mm512_reduce_add_pd(v); }

inline __m512d abs_pd(__m512d v) {
    const __m512i mask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL);
    return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v), mask));
}

inline void reduce_col_d(const double *col_ptr, std::size_t k, double &amax_out, double &vnrm_out) {
    __m512d vmax  = _mm512_setzero_pd();
    __m512d vsum  = _mm512_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= k; i += 8) {
        __m512d v  = load_pd(col_ptr + i);
        __m512d va = abs_pd(v);
        vmax       = _mm512_max_pd(vmax, va);
        vsum = _mm512_fmadd_pd(va, va, vsum);
    }
    __mmask8 tail_m = (1 << (k - i)) - 1;
    if (tail_m) {
        __m512d v  = _mm512_maskz_loadu_pd(tail_m, col_ptr + i);
        __m512d va = abs_pd(v);
        vmax       = _mm512_mask_max_pd(vmax, tail_m, vmax, va);
        vsum       = _mm512_mask_fmadd_pd(va, tail_m, va, vsum);
    }
    amax_out      = hmax_pd(vmax);
    double vsum_s = hsum_pd(vsum);
    if (vsum_s > 0) vsum_s = std::nextafter(vsum_s, INFINITY);
    vnrm_out = vsum_s;
}

inline void reduce_rows16_d(const double *base, std::size_t ld, std::size_t k,
                            double amax[16], double vnrm[16]) {
    __m512d vmax_lo = _mm512_setzero_pd(), vsum_lo = _mm512_setzero_pd();
    __m512d vmax_hi = _mm512_setzero_pd(), vsum_hi = _mm512_setzero_pd();
    for (std::size_t j = 0; j < k; ++j) {
        const double *row_j = base + j * ld;
        __m512d       v_lo  = _mm512_loadu_pd(row_j);
        __m512d       v_hi  = _mm512_loadu_pd(row_j + 8);
        __m512d       a_lo  = abs_pd(v_lo);
        __m512d       a_hi  = abs_pd(v_hi);
        vmax_lo             = _mm512_max_pd(vmax_lo, a_lo);
        vmax_hi             = _mm512_max_pd(vmax_hi, a_hi);
        vsum_lo             = _mm512_fmadd_pd(a_lo, a_lo, vsum_lo);
        vsum_hi             = _mm512_fmadd_pd(a_hi, a_hi, vsum_hi);
    }
    _mm512_storeu_pd(amax, vmax_lo);
    _mm512_storeu_pd(amax + 8, vmax_hi);
    double tmp[16];
    _mm512_storeu_pd(tmp, vsum_lo);
    _mm512_storeu_pd(tmp + 8, vsum_hi);
    for (int i = 0; i < 16; ++i) vnrm[i] = (tmp[i] > 0) ? std::nextafter(tmp[i], INFINITY) : tmp[i];
}

template <unsigned num_moduli>
inline void emit_scaled_col_int32_d(const double *col_ptr, std::size_t k, int sft,
                                    std::int8_t *X_lo_base, std::size_t k_pad,
                                    std::size_t inc_lo) {
    const __m512d sft_v = _mm512_set1_pd((double)sft);

    struct BarrettConst {
        int32_t p;
        int32_t p_inv;
    };
    auto make_bar = [](int idx) {
        int32_t p      = table::moduli[idx];
        int32_t p_inv  = (int32_t)(4294967296ULL / (uint64_t)p);
        return BarrettConst{p, p_inv};
    };

    std::size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512d v_lo = _mm512_loadu_pd(col_ptr + i);
        __m512d v_hi = _mm512_loadu_pd(col_ptr + i + 8);
        __m512d s_lo = _mm512_scalef_pd(v_lo, sft_v);
        __m512d s_hi = _mm512_scalef_pd(v_hi, sft_v);
        __m256i i32_lo = _mm512_cvttpd_epi32(s_lo);
        __m256i i32_hi = _mm512_cvttpd_epi32(s_hi);
        __m512i i32 = _mm512_inserti32x8(_mm512_castsi256_si512(i32_lo), i32_hi, 1);

        std::int8_t *out_col = X_lo_base + i;
        for (unsigned j = 0; j < num_moduli; ++j) {
            __m512i rem;
            if (j == 0) {
                rem = detail::mod_256_epi32(i32);
            } else {
                BarrettConst bc = make_bar(j);
                rem             = detail::mod_p_epi32(i32, bc.p, bc.p_inv);
            }
            __m128i packed = detail::pack_epi32_to_epi8(rem);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out_col), packed);
            out_col += inc_lo;
        }
    }
    for (; i < k; ++i) {
        double       x   = col_ptr[i];
        double       scl = std::scalbn(x, sft);
        int32_t      a   = (int32_t)scl;
        std::int8_t *out = X_lo_base + i;
        for (unsigned j = 0; j < num_moduli; ++j) {
            int32_t p = table::moduli[j];
            int32_t r;
            if (j == 0) {
                r = (int8_t)(a & 0xFF);
            } else {
                r = a % p;
                if (r > p / 2) r -= p;
                else if (r < -p / 2)
                    r += p;
            }
            out[0] = (std::int8_t)r;
            out += inc_lo;
        }
    }
}

struct FpME {
    std::int64_t mant;
    int          exp;
};

inline FpME make_fp_mant_exp_d(double a, int sft) {
    FpME r;
    const std::uint64_t bits = __builtin_bit_cast(std::uint64_t, a);
    const std::uint64_t sign = bits >> 63;
    int                 eraw = int((bits >> 52) & 0x7FFull);
    const std::uint64_t frac = bits & ((1ull << 52) - 1ull);
    if (eraw == 0 && frac == 0) { r.mant = 0; r.exp = 0; return r; }

    const int unbiased = eraw - 1023 + sft;
    const std::uint64_t sig = (eraw == 0) ? frac : ((1ull << 52) | frac);

    const int out_exp = (unbiased - 62 > 0) ? (unbiased - 62) : 0;

    std::int64_t mant;
    if (unbiased > 62) {
        mant = (std::int64_t)(sig << 10);
    } else {
        const int sh = unbiased - 52;
        const std::uint64_t mag = (sh >= 0) ? (sig << sh) : (sig >> (-sh));
        mant = (std::int64_t)mag;
    }
    r.exp  = out_exp;
    r.mant = sign ? -mant : mant;
    return r;
}

inline std::int32_t mod_int64_p(std::int64_t a, std::int32_t p) {
    std::int64_t rr = a % p;
    std::int32_t r  = (std::int32_t)rr;
    if (r > p / 2) r -= p;
    else if (r < -p / 2)
        r += p;
    return r;
}

template <unsigned num_moduli>
inline void emit_scaled_one_int64_d(double x, int sft,
                                    std::int8_t *out,
                                    std::size_t inc_lo) {
    double  scl = std::scalbn(x, sft);
    int64_t a   = (int64_t)scl;

    for (unsigned j = 0; j < num_moduli; ++j) {
        int32_t p = table::moduli[j];
        int32_t r;
        if (j == 0) {
            r = (int8_t)(a & 0xFF);
        } else {
            r = mod_int64_p(a, p);
        }
        out[0] = (std::int8_t)r;
        out += inc_lo;
    }
}

template <unsigned num_moduli>
inline void emit_scaled_col_fp_d(const double *col_ptr, std::size_t k, int sft,
                                 std::int8_t *X_lo_base, std::size_t k_pad,
                                 std::size_t inc_lo) {
    for (std::size_t i = 0; i < k; ++i) {
        FpME          d   = make_fp_mant_exp_d(col_ptr[i], sft);
        std::int8_t  *out = X_lo_base + i;
        for (unsigned j = 0; j < num_moduli; ++j) {
            std::int32_t p = table::moduli[j];
            std::int32_t r;
            if (j == 0) {
                if (d.exp >= 8) {
                    r = 0;
                } else {
                    std::int64_t v = (d.mant << d.exp) & 0xFFLL;
                    r              = (std::int8_t)(v & 0xFF);
                }
            } else {
                std::int32_t rem1 = mod_int64_p(d.mant, p);
                std::int32_t rem2 = table::get_mod_pow2(j, d.exp);
                std::int64_t prod = (std::int64_t)rem1 * (std::int64_t)rem2;
                r                 = mod_int64_p(prod, p);
            }
            out[0] = (std::int8_t)r;
            out += inc_lo;
        }
    }
}

template <unsigned num_moduli>
inline void emit_scaled_col_int64_d(const double *col_ptr, std::size_t k, int sft,
                                    std::int8_t *X_lo_base, std::size_t k_pad,
                                    std::size_t inc_lo) {
    for (std::size_t i = 0; i < k; ++i) {
        emit_scaled_one_int64_d<num_moduli>(col_ptr[i], sft, X_lo_base + i, inc_lo);
    }
}

template <unsigned num_moduli>
inline void emit_scaled_col_fpmod_d(const double *col_ptr, std::size_t k, int sft,
                                    std::int8_t *X_lo_base, std::size_t k_pad,
                                    std::size_t inc_lo) {
    const __m512d sft_v       = _mm512_set1_pd((double)sft);
    const __m512d exact_limit = _mm512_set1_pd(0x1.0p52);
    const __m512i abs_mask_i  = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL);

    std::size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512d v_lo = _mm512_loadu_pd(col_ptr + i);
        __m512d v_hi = _mm512_loadu_pd(col_ptr + i + 8);
        __m512d s_lo = _mm512_scalef_pd(v_lo, sft_v);
        __m512d s_hi = _mm512_scalef_pd(v_hi, sft_v);

        __m512d a_lo = _mm512_roundscale_pd(s_lo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m512d a_hi = _mm512_roundscale_pd(s_hi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

        __m512d abs_lo = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a_lo), abs_mask_i));
        __m512d abs_hi = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a_hi), abs_mask_i));
        __mmask8 safe_lo = _mm512_cmp_pd_mask(abs_lo, exact_limit, _CMP_LT_OQ);
        __mmask8 safe_hi = _mm512_cmp_pd_mask(abs_hi, exact_limit, _CMP_LT_OQ);
        if ((safe_lo != 0xFFu) || (safe_hi != 0xFFu)) {
            for (int t = 0; t < 16; ++t) {
                emit_scaled_one_int64_d<num_moduli>(col_ptr[i + t], sft,
                                                    X_lo_base + i + (std::size_t)t,
                                                    inc_lo);
            }
            continue;
        }

        std::int8_t *out_col = X_lo_base + i;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const double pj = (double)table::moduli[j];
            __m512d p_v     = _mm512_set1_pd(pj);
            __m512d half_v  = _mm512_set1_pd(0.5 * pj);

            __m512d q_lo = _mm512_roundscale_pd(_mm512_div_pd(a_lo, p_v),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512d q_hi = _mm512_roundscale_pd(_mm512_div_pd(a_hi, p_v),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512d r_lo = _mm512_fnmadd_pd(q_lo, p_v, a_lo);
            __m512d r_hi = _mm512_fnmadd_pd(q_hi, p_v, a_hi);

            __mmask8 hi_lo = _mm512_cmp_pd_mask(r_lo, half_v, _CMP_GT_OQ);
            __mmask8 hi_hi = _mm512_cmp_pd_mask(r_hi, half_v, _CMP_GT_OQ);
            r_lo = _mm512_mask_sub_pd(r_lo, hi_lo, r_lo, p_v);
            r_hi = _mm512_mask_sub_pd(r_hi, hi_hi, r_hi, p_v);

            __m512d neg_half_v = _mm512_sub_pd(_mm512_setzero_pd(), half_v);
            __mmask8 lo_lo = _mm512_cmp_pd_mask(r_lo, neg_half_v, _CMP_LT_OQ);
            __mmask8 lo_hi = _mm512_cmp_pd_mask(r_hi, neg_half_v, _CMP_LT_OQ);
            r_lo = _mm512_mask_add_pd(r_lo, lo_lo, r_lo, p_v);
            r_hi = _mm512_mask_add_pd(r_hi, lo_hi, r_hi, p_v);

            __m256i lo32  = _mm512_cvttpd_epi32(r_lo);
            __m256i hi32  = _mm512_cvttpd_epi32(r_hi);
            __m512i rem32 = _mm512_inserti32x8(_mm512_castsi256_si512(lo32), hi32, 1);
            __m128i packed = detail::pack_epi32_to_epi8(rem32);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out_col), packed);
            out_col += inc_lo;
        }
    }

    for (; i < k; ++i) {
        emit_scaled_one_int64_d<num_moduli>(col_ptr[i], sft, X_lo_base + i, inc_lo);
    }
}

template <unsigned num_moduli>
void scale_double(bool     cols_major,
                  const double *X,
                  std::size_t ld_x,
                  std::size_t k,
                  std::size_t n_outer,
                  std::int8_t *X_lo,
                  std::size_t  k_pad,
                  std::size_t  inc_lo,
                  std::int16_t *sft_out) {
    const float log2P = table::getLog2P_f(num_moduli);

    if (cols_major) {
#pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < n_outer; ++c) {
            const double *col = X + c * ld_x;
            double amax, vnrm;
            reduce_col_d(col, k, amax, vnrm);
            int sft     = detail::compute_sft_fast_d(amax, vnrm, log2P);
            sft_out[c] = (std::int16_t)(-sft);

            std::int8_t *out_base = X_lo + c * k_pad;
            if constexpr (num_moduli <= detail::Threshold::S) {
                emit_scaled_col_int32_d<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            } else if constexpr (num_moduli <= detail::Threshold::M) {
                emit_scaled_col_fpmod_d<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            } else {
                emit_scaled_col_fp_d<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            }
            for (unsigned j = 0; j < num_moduli; ++j) {
                std::int8_t *tail = out_base + j * inc_lo + k;
                std::memset(tail, 0, k_pad - k);
            }
        }
    } else {
        std::size_t n_rows = n_outer;
#pragma omp parallel for schedule(static)
        for (std::size_t r_base = 0; r_base < n_rows; r_base += 16) {
            std::size_t   r_end = std::min<std::size_t>(r_base + 16, n_rows);
            const double *base  = X + r_base;

            double amax[16], vnrm[16];
            if (r_end - r_base == 16) {
                reduce_rows16_d(base, ld_x, k, amax, vnrm);
            } else {
                for (std::size_t r = r_base; r < r_end; ++r) {
                    __m512d vmax = _mm512_setzero_pd(), vsum = _mm512_setzero_pd();
                    std::size_t j = 0;
                    for (; j + 8 <= k; j += 8) {
                        alignas(64) double tmp[8];
                        for (int t = 0; t < 8; ++t) tmp[t] = X[(j + t) * ld_x + r];
                        __m512d v  = _mm512_load_pd(tmp);
                        __m512d va = abs_pd(v);
                        vmax       = _mm512_max_pd(vmax, va);
                        vsum       = _mm512_fmadd_pd(va, va, vsum);
                    }
                    double vmx = hmax_pd(vmax), vs = hsum_pd(vsum);
                    for (; j < k; ++j) {
                        double x = X[j * ld_x + r];
                        double a = std::fabs(x);
                        vmx      = std::fmax(vmx, a);
                        vs += a * a;
                    }
                    amax[r - r_base] = vmx;
                    vnrm[r - r_base] = (vs > 0) ? std::nextafter(vs, INFINITY) : vs;
                }
            }
            for (std::size_t r = r_base; r < r_end; ++r) {
                int lane   = (int)(r - r_base);
                int sft    = detail::compute_sft_fast_d(amax[lane], vnrm[lane], log2P);
                sft_out[r] = (std::int16_t)(-sft);

                std::int8_t *out_base = X_lo + r * k_pad;
                alignas(64) double buf[4096];
                std::size_t        chunk = (k <= sizeof(buf) / sizeof(double)) ? k : sizeof(buf) / sizeof(double);
                std::size_t        off   = 0;
                while (off < k) {
                    std::size_t this_k = std::min(chunk, k - off);
                    for (std::size_t j = 0; j < this_k; ++j) {
                        buf[j] = X[(off + j) * ld_x + r];
                    }
                    if constexpr (num_moduli <= detail::Threshold::S) {
                        emit_scaled_col_int32_d<num_moduli>(buf, this_k, sft,
                                                            out_base + off, k_pad, inc_lo);
                    } else if constexpr (num_moduli <= detail::Threshold::M) {
                        emit_scaled_col_fpmod_d<num_moduli>(buf, this_k, sft,
                                                            out_base + off, k_pad, inc_lo);
                    } else {
                        emit_scaled_col_fp_d<num_moduli>(buf, this_k, sft,
                                                         out_base + off, k_pad, inc_lo);
                    }
                    off += this_k;
                }
                for (unsigned j = 0; j < num_moduli; ++j) {
                    std::int8_t *tail = out_base + j * inc_lo + k;
                    std::memset(tail, 0, k_pad - k);
                }
            }
        }
    }
}

inline void reduce_col_f(const float *col_ptr, std::size_t k, float &amax_out, float &vnrm_out) {
    __m512 vmax = _mm512_setzero_ps();
    __m512 vsum = _mm512_setzero_ps();
    std::size_t i = 0;
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    for (; i + 16 <= k; i += 16) {
        __m512 v  = _mm512_loadu_ps(col_ptr + i);
        __m512 va = _mm512_and_ps(v, sign_mask);
        vmax      = _mm512_max_ps(vmax, va);
        vsum      = _mm512_fmadd_ps(va, va, vsum);
    }
    __mmask16 tm = (1U << (k - i)) - 1;
    if (tm) {
        __m512 v  = _mm512_maskz_loadu_ps(tm, col_ptr + i);
        __m512 va = _mm512_and_ps(v, sign_mask);
        vmax      = _mm512_mask_max_ps(vmax, tm, vmax, va);
        vsum      = _mm512_mask_fmadd_ps(va, tm, va, vsum);
    }
    amax_out = _mm512_reduce_max_ps(vmax);
    float vs = _mm512_reduce_add_ps(vsum);
    if (vs > 0) vs = std::nextafterf(vs, INFINITY);
    vnrm_out = vs;
}

template <unsigned num_moduli>
inline void emit_scaled_col_int32_f(const float *col_ptr, std::size_t k, int sft,
                                    std::int8_t *X_lo_base, std::size_t k_pad,
                                    std::size_t inc_lo) {
    const __m512 sft_v = _mm512_set1_ps((float)sft);
    std::size_t  i     = 0;
    for (; i + 16 <= k; i += 16) {
        __m512  v   = _mm512_loadu_ps(col_ptr + i);
        __m512  s   = _mm512_scalef_ps(v, sft_v);
        __m512i i32 = _mm512_cvttps_epi32(s);

        std::int8_t *out_col = X_lo_base + i;
        for (unsigned j = 0; j < num_moduli; ++j) {
            __m512i rem;
            if (j == 0) {
                rem = detail::mod_256_epi32(i32);
            } else {
                int32_t p     = table::moduli[j];
                int32_t p_inv = (int32_t)(4294967296ULL / (uint64_t)p);
                rem           = detail::mod_p_epi32(i32, p, p_inv);
            }
            __m128i packed = detail::pack_epi32_to_epi8(rem);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out_col), packed);
            out_col += inc_lo;
        }
    }
    for (; i < k; ++i) {
        float   x = col_ptr[i];
        float   s = std::scalbnf(x, sft);
        int32_t a = (int32_t)s;
        std::int8_t *out = X_lo_base + i;
        for (unsigned j = 0; j < num_moduli; ++j) {
            int32_t p = table::moduli[j];
            int32_t r;
            if (j == 0) {
                r = (int8_t)(a & 0xFF);
            } else {
                r = a % p;
                if (r > p / 2) r -= p;
                else if (r < -p / 2)
                    r += p;
            }
            out[0] = (std::int8_t)r;
            out += inc_lo;
        }
    }
}

template <unsigned num_moduli>
void scale_float(bool     cols_major,
                 const float *X,
                 std::size_t ld_x,
                 std::size_t k,
                 std::size_t n_outer,
                 std::int8_t *X_lo,
                 std::size_t  k_pad,
                 std::size_t  inc_lo,
                 std::int16_t *sft_out) {
    const float log2P = table::getLog2P_f(num_moduli);
    if (cols_major) {
#pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < n_outer; ++c) {
            const float *col = X + c * ld_x;
            float        amax, vnrm;
            reduce_col_f(col, k, amax, vnrm);
            int sft     = detail::compute_sft_fast_f(amax, vnrm, log2P);
            sft_out[c] = (std::int16_t)(-sft);
            std::int8_t *out_base = X_lo + c * k_pad;
            emit_scaled_col_int32_f<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            for (unsigned j = 0; j < num_moduli; ++j) {
                std::memset(out_base + j * inc_lo + k, 0, k_pad - k);
            }
        }
    } else {
        // Strided path: pack 16 rows via explicit gather.
        std::size_t n_rows = n_outer;
#pragma omp parallel for schedule(static)
        for (std::size_t r = 0; r < n_rows; ++r) {
            alignas(64) float buf[65536 / 4]; // cap
            std::size_t        off = 0;
            float              amax_acc = 0, vnrm_acc = 0;
            while (off < k) {
                std::size_t this_k = std::min<std::size_t>(sizeof(buf) / sizeof(float), k - off);
                for (std::size_t j = 0; j < this_k; ++j) buf[j] = X[(off + j) * ld_x + r];
                float amax, vnrm;
                reduce_col_f(buf, this_k, amax, vnrm);
                amax_acc = std::fmax(amax_acc, amax);
                vnrm_acc += vnrm;
                off += this_k;
            }
            int sft     = detail::compute_sft_fast_f(amax_acc, vnrm_acc, log2P);
            sft_out[r] = (std::int16_t)(-sft);

            std::int8_t *out_base = X_lo + r * k_pad;
            off                   = 0;
            while (off < k) {
                std::size_t this_k = std::min<std::size_t>(sizeof(buf) / sizeof(float), k - off);
                for (std::size_t j = 0; j < this_k; ++j) buf[j] = X[(off + j) * ld_x + r];
                emit_scaled_col_int32_f<num_moduli>(buf, this_k, sft, out_base + off, k_pad, inc_lo);
                off += this_k;
            }
            for (unsigned j = 0; j < num_moduli; ++j) {
                std::memset(out_base + j * inc_lo + k, 0, k_pad - k);
            }
        }
    }
}

inline void emit_high_order_int8_d(const double *col_ptr, std::size_t k, int sft,
                                   std::int8_t *X_lo_high, std::size_t k_pad) {
    const __m512d sft_v = _mm512_set1_pd((double)sft);
    const __m512i abs_mask_i = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL);
    std::size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512d v_lo = _mm512_loadu_pd(col_ptr + i);
        __m512d v_hi = _mm512_loadu_pd(col_ptr + i + 8);
        // |x|
        __m512d a_lo = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v_lo), abs_mask_i));
        __m512d a_hi = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v_hi), abs_mask_i));
        // 2^sft * |x|
        __m512d s_lo = _mm512_scalef_pd(a_lo, sft_v);
        __m512d s_hi = _mm512_scalef_pd(a_hi, sft_v);
        // ceil
        __m512d c_lo = _mm512_roundscale_pd(s_lo, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        __m512d c_hi = _mm512_roundscale_pd(s_hi, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        __m256i i32_lo = _mm512_cvttpd_epi32(c_lo);
        __m256i i32_hi = _mm512_cvttpd_epi32(c_hi);
        __m512i i32 = _mm512_inserti32x8(_mm512_castsi256_si512(i32_lo), i32_hi, 1);
        __m128i packed = _mm512_cvtepi32_epi8(i32);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(X_lo_high + i), packed);
    }
    for (; i < k; ++i) {
        double a = std::fabs(col_ptr[i]);
        double s = std::scalbn(a, sft);
        double c = std::ceil(s);
        int    v = (int)c;
        if (v > 127) v = 127;
        X_lo_high[i] = (std::int8_t)v;
    }
}

template <typename T>
inline void extract_high_order_double(bool               cols_major,
                                      const T           *X,
                                      std::size_t        ldx,
                                      std::size_t        k,
                                      std::size_t        n_outer,
                                      std::int8_t       *X_lo_high,
                                      std::size_t        k_pad,
                                      std::int16_t      *sft_out) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "extract_high_order_double is generic over float/double");
    constexpr int maxUFP = detail::kMaxUFP_INT8;

    if (cols_major) {
#pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < n_outer; ++c) {
            const T *col = X + c * ldx;
            // scan amax
            double amax = 0;
            std::size_t j = 0;
            if constexpr (std::is_same_v<T, double>) {
                __m512d vm = _mm512_setzero_pd();
                const __m512i abs_i = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL);
                for (; j + 8 <= k; j += 8) {
                    __m512d v  = _mm512_loadu_pd(col + j);
                    __m512d a  = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v), abs_i));
                    vm         = _mm512_max_pd(vm, a);
                }
                amax = hmax_pd(vm);
                for (; j < k; ++j) amax = std::fmax(amax, std::fabs((double)col[j]));
            } else {
                for (; j < k; ++j) amax = std::fmax(amax, std::fabs((double)col[j]));
            }
            int sft = (amax == 0) ? 0 : (maxUFP - (int)std::ilogb(amax));
            sft_out[c] = (std::int16_t)sft;

            std::int8_t *out = X_lo_high + c * k_pad;
            if constexpr (std::is_same_v<T, double>) {
                emit_high_order_int8_d(col, k, sft, out, k_pad);
            } else {
                // scalar fallback for float input (not the hot path).
                for (std::size_t kk = 0; kk < k; ++kk) {
                    double a = std::fabs((double)col[kk]);
                    int    v = (int)std::ceil(std::scalbn(a, sft));
                    if (v > 127) v = 127;
                    out[kk] = (std::int8_t)v;
                }
            }
            std::memset(out + k, 0, k_pad - k);
        }
    } else {
        std::size_t n_rows = n_outer;
#pragma omp parallel for schedule(static)
        for (std::size_t r = 0; r < n_rows; ++r) {
            alignas(64) double buf[4096];
            std::size_t        off = 0;
            double             amax = 0;
            while (off < k) {
                std::size_t this_k = std::min<std::size_t>(sizeof(buf) / sizeof(double), k - off);
                for (std::size_t j = 0; j < this_k; ++j) buf[j] = (double)X[(off + j) * ldx + r];
                for (std::size_t j = 0; j < this_k; ++j) amax = std::fmax(amax, std::fabs(buf[j]));
                off += this_k;
            }
            int sft = (amax == 0) ? 0 : (maxUFP - (int)std::ilogb(amax));
            sft_out[r] = (std::int16_t)sft;
            std::int8_t *out = X_lo_high + r * k_pad;
            off = 0;
            while (off < k) {
                std::size_t this_k = std::min<std::size_t>(sizeof(buf) / sizeof(double), k - off);
                for (std::size_t j = 0; j < this_k; ++j) buf[j] = (double)X[(off + j) * ldx + r];
                emit_high_order_int8_d(buf, this_k, sft, out + off, k_pad);
                off += this_k;
            }
            std::memset(out + k, 0, k_pad - k);
        }
    }
}

template <unsigned num_moduli>
void scale_double_given_sft(bool                cols_major,
                            const double       *X,
                            std::size_t         ld_x,
                            std::size_t         k,
                            std::size_t         n_outer,
                            std::int8_t        *X_lo,
                            std::size_t         k_pad,
                            std::size_t         inc_lo,
                            const std::int16_t *neg_sft_in) {
    if (cols_major) {
#pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < n_outer; ++c) {
            const double *col = X + c * ld_x;
            int           sft = -(int)neg_sft_in[c];
            std::int8_t  *out_base = X_lo + c * k_pad;
            if constexpr (num_moduli <= detail::Threshold::S) {
                emit_scaled_col_int32_d<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            } else if constexpr (num_moduli <= detail::Threshold::M) {
                emit_scaled_col_int64_d<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            } else {
                emit_scaled_col_fp_d<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            }
            for (unsigned j = 0; j < num_moduli; ++j) {
                std::memset(out_base + j * inc_lo + k, 0, k_pad - k);
            }
        }
    } else {
        std::size_t n_rows = n_outer;
#pragma omp parallel for schedule(static)
        for (std::size_t r = 0; r < n_rows; ++r) {
            int sft = -(int)neg_sft_in[r];
            std::int8_t *out_base = X_lo + r * k_pad;
            alignas(64) double buf[4096];
            std::size_t chunk = sizeof(buf) / sizeof(double);
            std::size_t off   = 0;
            while (off < k) {
                std::size_t this_k = std::min(chunk, k - off);
                for (std::size_t j = 0; j < this_k; ++j) buf[j] = X[(off + j) * ld_x + r];
                if constexpr (num_moduli <= detail::Threshold::S) {
                    emit_scaled_col_int32_d<num_moduli>(buf, this_k, sft, out_base + off, k_pad, inc_lo);
                } else if constexpr (num_moduli <= detail::Threshold::M) {
                    emit_scaled_col_int64_d<num_moduli>(buf, this_k, sft, out_base + off, k_pad, inc_lo);
                } else {
                    emit_scaled_col_fp_d<num_moduli>(buf, this_k, sft, out_base + off, k_pad, inc_lo);
                }
                off += this_k;
            }
            for (unsigned j = 0; j < num_moduli; ++j) {
                std::memset(out_base + j * inc_lo + k, 0, k_pad - k);
            }
        }
    }
}

template <unsigned num_moduli>
void scale_float_given_sft(bool                cols_major,
                           const float        *X,
                           std::size_t         ld_x,
                           std::size_t         k,
                           std::size_t         n_outer,
                           std::int8_t        *X_lo,
                           std::size_t         k_pad,
                           std::size_t         inc_lo,
                           const std::int16_t *neg_sft_in) {
    if (cols_major) {
#pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < n_outer; ++c) {
            const float *col = X + c * ld_x;
            int           sft      = -(int)neg_sft_in[c];
            std::int8_t  *out_base = X_lo + c * k_pad;
            emit_scaled_col_int32_f<num_moduli>(col, k, sft, out_base, k_pad, inc_lo);
            for (unsigned j = 0; j < num_moduli; ++j)
                std::memset(out_base + j * inc_lo + k, 0, k_pad - k);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (std::size_t r = 0; r < n_outer; ++r) {
            int           sft      = -(int)neg_sft_in[r];
            std::int8_t  *out_base = X_lo + r * k_pad;
            alignas(64) float buf[65536 / 4];
            std::size_t chunk = sizeof(buf) / sizeof(float);
            std::size_t off   = 0;
            while (off < k) {
                std::size_t this_k = std::min<std::size_t>(chunk, k - off);
                for (std::size_t j = 0; j < this_k; ++j) buf[j] = X[(off + j) * ld_x + r];
                emit_scaled_col_int32_f<num_moduli>(buf, this_k, sft, out_base + off, k_pad, inc_lo);
                off += this_k;
            }
            for (unsigned j = 0; j < num_moduli; ++j)
                std::memset(out_base + j * inc_lo + k, 0, k_pad - k);
        }
    }
}

} // namespace scaling
} // namespace gemmul8_amx
