#pragma once
#include <cstddef>
#include <cstdint>

namespace gemmul8_avx512 {
namespace detail {

inline constexpr std::size_t kAlign = 64;

inline constexpr std::size_t kKPad = 64;

inline constexpr std::size_t kMPad = 16;
inline constexpr std::size_t kNPad = 16;

inline std::size_t padUp(std::size_t x, std::size_t mult) {
    return (x + mult - 1) / mult * mult;
}

inline void *alignPtr(void *p, std::size_t a = kAlign) {
    auto x = reinterpret_cast<std::uintptr_t>(p);
    x      = (x + (a - 1)) & ~(a - 1);
    return reinterpret_cast<void *>(x);
}

struct Threshold {
    static constexpr int S = 7;
    static constexpr int M = 15;
    static constexpr int L = 25;
};

} // namespace detail
} // namespace gemmul8_avx512
