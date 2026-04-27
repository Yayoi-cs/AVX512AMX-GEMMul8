#pragma once
#include "table.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace gemmul8_avx512 {
namespace invscal {

template <unsigned num_moduli>
inline void crt_dd_16lanes(const std::int8_t *const *c_ptr_base,
                           std::size_t              m,
                           std::size_t              ld_ct,
                           std::size_t              n_base,
                           bool                     use_dd,
                           double                   invP,
                           double                   P_hi,
                           double                   P_lo,
                           __m512d                 &out_lo,
                           __m512d                 &out_hi) {
    __m512d acc_lo = _mm512_setzero_pd();
    __m512d acc_hi = _mm512_setzero_pd();
    __m512d acc_lo2 = _mm512_setzero_pd(); 
    __m512d acc_hi2 = _mm512_setzero_pd();

    for (unsigned j = 0; j < num_moduli; ++j) {
        const std::int8_t *p = c_ptr_base[j] + n_base * ld_ct + m; 
        alignas(64) std::int8_t tmp[16];
        for (int i = 0; i < 16; ++i) tmp[i] = p[i * ld_ct];
        __m128i v_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tmp));
        __m512i v_i32 = _mm512_cvtepi8_epi32(v_i8);
        __m256i lo_i32 = _mm512_extracti32x8_epi32(v_i32, 0);
        __m256i hi_i32 = _mm512_extracti32x8_epi32(v_i32, 1);
        __m512d v_d_lo = _mm512_cvtepi32_pd(lo_i32);
        __m512d v_d_hi = _mm512_cvtepi32_pd(hi_i32);

        __m512d qpi_hi = _mm512_set1_pd(table::getQPi1(num_moduli, j));
        acc_lo         = _mm512_fmadd_pd(qpi_hi, v_d_lo, acc_lo);
        acc_hi         = _mm512_fmadd_pd(qpi_hi, v_d_hi, acc_hi);

        if (use_dd) {
            table::DD dd = table::getQPi2(num_moduli, j);
            __m512d  qd  = _mm512_set1_pd(dd.hi);
            __m512d  ql  = _mm512_set1_pd(dd.lo);
            acc_lo  = _mm512_fmadd_pd(qd, v_d_lo, (j == 0) ? _mm512_setzero_pd() : acc_lo);
            acc_hi  = _mm512_fmadd_pd(qd, v_d_hi, (j == 0) ? _mm512_setzero_pd() : acc_hi);
            acc_lo2 = _mm512_fmadd_pd(ql, v_d_lo, (j == 0) ? _mm512_setzero_pd() : acc_lo2);
            acc_hi2 = _mm512_fmadd_pd(ql, v_d_hi, (j == 0) ? _mm512_setzero_pd() : acc_hi2);
        }
    }

    __m512d invP_v = _mm512_set1_pd(invP);
    __m512d Phi_v  = _mm512_set1_pd(P_hi);
    __m512d Plo_v  = _mm512_set1_pd(P_lo);

    if (use_dd) {
        __m512d quot_lo  = _mm512_mul_pd(invP_v, acc_lo);
        __m512d quot_hi  = _mm512_mul_pd(invP_v, acc_hi);
        quot_lo          = _mm512_roundscale_pd(quot_lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        quot_hi          = _mm512_roundscale_pd(quot_hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512d t_lo  = _mm512_fmadd_pd(Phi_v, quot_lo, acc_lo);
        __m512d t_hi  = _mm512_fmadd_pd(Phi_v, quot_hi, acc_hi);
        t_lo          = _mm512_add_pd(t_lo, acc_lo2);
        t_hi          = _mm512_add_pd(t_hi, acc_hi2);
        out_lo        = _mm512_fmadd_pd(Plo_v, quot_lo, t_lo);
        out_hi        = _mm512_fmadd_pd(Plo_v, quot_hi, t_hi);
    } else {
        __m512d quot_lo = _mm512_mul_pd(invP_v, acc_lo);
        __m512d quot_hi = _mm512_mul_pd(invP_v, acc_hi);
        quot_lo         = _mm512_roundscale_pd(quot_lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        quot_hi         = _mm512_roundscale_pd(quot_hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        out_lo          = _mm512_fmadd_pd(Phi_v, quot_lo, acc_lo);
        out_hi          = _mm512_fmadd_pd(Phi_v, quot_hi, acc_hi);
    }
}

template <unsigned num_moduli>
inline double crt_scalar(const std::int8_t *const *c_ptr,
                         std::size_t              m,
                         std::size_t              ld_ct,
                         std::size_t              n,
                         bool                     use_dd,
                         double                   invP,
                         double                   P_hi,
                         double                   P_lo) {
    if (!use_dd) {
        double acc = 0;
        for (unsigned j = 0; j < num_moduli; ++j) {
            int8_t v = c_ptr[j][n * ld_ct + m];
            acc = std::fma(table::getQPi1(num_moduli, j), (double)v, acc);
        }
        double quot = std::rint(invP * acc);
        return std::fma(P_hi, quot, acc);
    } else {
        double acc_hi = 0, acc_lo = 0;
        for (unsigned j = 0; j < num_moduli; ++j) {
            int8_t    v  = c_ptr[j][n * ld_ct + m];
            table::DD dd = table::getQPi2(num_moduli, j);
            acc_hi       = std::fma(dd.hi, (double)v, acc_hi);
            acc_lo       = std::fma(dd.lo, (double)v, acc_lo);
        }
        double quot = std::rint(invP * acc_hi);
        double t    = std::fma(P_hi, quot, acc_hi) + acc_lo;
        return std::fma(P_lo, quot, t);
    }
}

template <typename T, unsigned num_moduli>
void inverse_scale_and_writeback(const std::int8_t *const *C_mid_ptrs,
                                 std::size_t              ld_ct,
                                 std::size_t              m,
                                 std::size_t              n,
                                 T                        alpha,
                                 T                        beta,
                                 T                       *C,
                                 std::size_t              ldc,
                                 const std::int16_t      *sftA,
                                 const std::int16_t      *sftB) {
    constexpr bool use_dd = (num_moduli > table::kPisDouble);
    const double  invP    = table::getInvP(num_moduli);
    const double  P_hi    = table::getP_hi(num_moduli);
    const double  P_lo    = use_dd ? table::getP_lo(num_moduli) : 0.0;

#pragma omp parallel for schedule(static)
    for (std::size_t nn = 0; nn < n; ++nn) {
        for (std::size_t mm = 0; mm < m; ++mm) {
            double    acc_hi = 0, acc_lo = 0;
            for (unsigned j = 0; j < num_moduli; ++j) {
                std::int8_t v = C_mid_ptrs[j][nn * ld_ct + mm];
                if constexpr (use_dd) {
                    table::DD dd = table::getQPi2(num_moduli, j);
                    acc_hi       = std::fma(dd.hi, (double)v, acc_hi);
                    acc_lo       = std::fma(dd.lo, (double)v, acc_lo);
                } else {
                    acc_hi = std::fma(table::getQPi1(num_moduli, j), (double)v, acc_hi);
                }
            }
            double quot = std::rint(invP * acc_hi);
            double crt;
            if constexpr (use_dd) {
                double t = std::fma(P_hi, quot, acc_hi) + acc_lo;
                crt      = std::fma(P_lo, quot, t);
            } else {
                crt = std::fma(P_hi, quot, acc_hi);
            }
            int sft = (int)sftA[mm] + (int)sftB[nn];
            double val = std::scalbn(crt, sft);

            std::size_t idxC = nn * ldc + mm;
            T           cur  = C[idxC];
            T           ab   = (T)val;
            T           out  = (T)((double)alpha * (double)ab + (double)beta * (double)cur);
            C[idxC]          = out;
        }
    }
}

} // namespace invscal
} // namespace gemmul8_avx512
