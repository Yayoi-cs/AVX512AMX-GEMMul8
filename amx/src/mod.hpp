#pragma once
#include "math_helpers.hpp"
#include "table.hpp"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace gemmul8_amx {
namespace mod_pass {

inline void conv_hi2mid(unsigned    modulus_idx,
                        const std::int32_t *C_hi,
                        std::size_t N,
                        std::int8_t *C_mid) {
    if (modulus_idx == 0) {
        std::size_t i = 0;
        for (; i + 16 <= N; i += 16) {
            __m512i x      = _mm512_loadu_si512(C_hi + i);
            __m512i rem    = detail::mod_256_epi32(x);
            __m128i packed = detail::pack_epi32_to_epi8(rem);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(C_mid + i), packed);
        }
        for (; i < N; ++i) {
            C_mid[i] = (std::int8_t)(C_hi[i] & 0xFF);
        }
        return;
    }

    const int32_t p     = table::moduli[modulus_idx];
    const int32_t p_inv = (int32_t)(4294967296ULL / (uint64_t)p);

    std::size_t i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512i x      = _mm512_loadu_si512(C_hi + i);
        __m512i rem    = detail::mod_p_epi32(x, p, p_inv);
        __m128i packed = detail::pack_epi32_to_epi8(rem);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(C_mid + i), packed);
    }
    for (; i < N; ++i) {
        int32_t a = C_hi[i];
        int32_t r = a % p;
        if (r > p / 2) r -= p;
        else if (r < -p / 2)
            r += p;
        C_mid[i] = (std::int8_t)r;
    }
}

} // namespace mod_pass
} // namespace gemmul8_amx
