#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gemmul8_amx {

enum Op : char { N = 'N', T = 'T' };

template <typename T>
std::size_t workSize(std::size_t m,
                     std::size_t n,
                     std::size_t k,
                     unsigned num_moduli,
                     std::size_t *workSizeA = nullptr,
                     std::size_t *workSizeB = nullptr);

template <typename T>
std::vector<double> gemm(Op op_A,
                         Op op_B,
                         std::size_t m,
                         std::size_t n,
                         std::size_t k,
                         T alpha,
                         const T *A,
                         std::size_t lda,
                         const T *B,
                         std::size_t ldb,
                         T beta,
                         T *C,
                         std::size_t ldc,
                         unsigned num_moduli,
                         void *work,
                         bool fastmode = false);

} // namespace gemmul8_amx
