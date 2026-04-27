#pragma once
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace gemmul8_avx512 {
namespace detail {

inline int compute_sft_fast_d(double amax, double vnrm, float log2P) {
    int exponent      = (amax == 0.0) ? 0 : (int)std::ilogb(vnrm);
    double scaled_vnm = std::scalbn(vnrm, -exponent);
    float vnmf = (float)scaled_vnm;
    if ((double)vnmf < scaled_vnm) {
        vnmf = std::nextafterf(vnmf, INFINITY);
    }
    float log2vsum = std::log2f(vnmf) + (float)exponent;
    float log2vnrm = log2vsum * 0x1.0000060000000p-1f;
    float exp1     = log2P - 1.5f - std::fmax(1.0f, log2vnrm);
    int   sft_abs  = (int)std::floor(exp1);
    int   amax_exp = (amax == 0.0) ? 0 : (int)std::ilogb(amax);
    return sft_abs - amax_exp;
}

inline int compute_sft_fast_f(float amax, float vnrm, float log2P) {
    if (amax == 0.0f) return 0;
    float log2vsum = std::log2f(vnrm);
    float log2vnrm = log2vsum * 0x1.0000060000000p-1f;
    float exp1     = log2P - 1.5f - std::fmax(1.0f, log2vnrm);
    int   sft_abs  = (int)std::floor(exp1);
    int   amax_exp = (int)std::ilogb(amax);
    return sft_abs - amax_exp;
}

inline int compute_sft_accu(int32_t amax, float log2P) {
    if (amax <= 0) return 0;
    float log2amax = std::log2f((float)amax);
    __m128 m_vec = _mm_set_ss(-0x1.0000060000000p-1f);
    __m128 a_vec = _mm_set_ss(log2amax);
    __m128 p_vec = _mm_set_ss(log2P);
    __m128 r_vec = _mm_fmadd_round_ss(m_vec, a_vec, p_vec,
                                      _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    float  r     = _mm_cvtss_f32(r_vec);
    return (int)std::floor(r);
}

inline constexpr int kMaxUFP_INT8 = 5;

inline __m512i mod_p_epi32(__m512i a, int32_t p, int32_t p_inv_32) {
    const __m512i v_p       = _mm512_set1_epi32(p);
    const __m512i v_p_inv   = _mm512_set1_epi32(p_inv_32);
    const __m512i v_p_half  = _mm512_set1_epi32(p / 2);
    const __m512i v_p_mhalf = _mm512_set1_epi32(-(p / 2));

    __m512i a_odd     = _mm512_srli_epi64(a, 32);        
    __m512i prod_even = _mm512_mul_epi32(a, v_p_inv);    s
    __m512i prod_odd  = _mm512_mul_epi32(a_odd, v_p_inv);

    __m512i hi_even = _mm512_srai_epi64(prod_even, 32);
    __m512i hi_odd  = _mm512_srai_epi64(prod_odd, 32);

    __m512i hi_odd_up = _mm512_slli_epi64(hi_odd, 32);
    __m512i q         = _mm512_mask_blend_epi32(0xAAAAu, hi_even, hi_odd_up);

    __m512i rem = _mm512_sub_epi32(a, _mm512_mullo_epi32(v_p, q));

    __mmask16 hi = _mm512_cmpgt_epi32_mask(rem, v_p_half);
    rem          = _mm512_mask_sub_epi32(rem, hi, rem, v_p);
    __mmask16 lo = _mm512_cmplt_epi32_mask(rem, v_p_mhalf);
    rem          = _mm512_mask_add_epi32(rem, lo, rem, v_p);
    return rem;
}

inline __m512i mod_256_epi32(__m512i a) {
    __m512i low8 = _mm512_and_si512(a, _mm512_set1_epi32(0xFF));
    __m512i   mask = _mm512_set1_epi32(128);
    __mmask16 m    = _mm512_test_epi32_mask(low8, mask);
    return _mm512_mask_sub_epi32(low8, m, low8, _mm512_set1_epi32(256));
}

inline __m128i pack_epi32_to_epi8(__m512i x) {
    return _mm512_cvtepi32_epi8(x);
}

} // namespace detail
} // namespace gemmul8_avx512
