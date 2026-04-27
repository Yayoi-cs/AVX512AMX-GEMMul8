#include "../include/gemmul8_avx512.hpp"

#include "common.hpp"
#include "inverse_scaling.hpp"
#include "matmult_vnni.hpp"
#include "mod.hpp"
#include "scaling.hpp"
#include "table.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gemmul8_avx512 {

namespace {

struct Layout {
    std::size_t k_pad;
    std::size_t m_pad;
    std::size_t n_pad;
    std::size_t a_lo_bytes; 
    std::size_t b_lo_bytes; 
    std::size_t a_pk_bytes; 
    std::size_t c_mid_bytes;
    std::size_t c_hi_bytes;
    std::size_t sft_a_bytes;
    std::size_t sft_b_bytes;
    std::size_t s_b_bytes;
    std::size_t a_lo_high_bytes; 
    std::size_t b_lo_high_bytes;
    std::size_t a_pk_high_bytes;
    std::size_t total_bytes;
};

inline Layout compute_layout(std::size_t m, std::size_t n, std::size_t k,
                             unsigned    num_moduli) {
    Layout lo{};
    lo.k_pad        = detail::padUp(k, detail::kKPad);
    lo.m_pad        = detail::padUp(m, 16);  
    lo.n_pad        = detail::padUp(n, detail::kNPad);
    lo.a_lo_bytes   = detail::padUp(m * lo.k_pad, detail::kAlign);
    lo.b_lo_bytes   = detail::padUp(n * lo.k_pad, detail::kAlign);
    lo.a_pk_bytes   = detail::padUp(lo.m_pad * lo.k_pad, detail::kAlign);
    lo.c_mid_bytes  = detail::padUp(m * n, detail::kAlign);
    lo.c_hi_bytes   = detail::padUp(lo.m_pad * lo.n_pad * sizeof(std::int32_t), detail::kAlign);
    lo.sft_a_bytes  = detail::padUp(m * sizeof(std::int16_t), detail::kAlign);
    lo.sft_b_bytes  = detail::padUp(n * sizeof(std::int16_t), detail::kAlign);
    lo.s_b_bytes    = detail::padUp(lo.n_pad * sizeof(std::int32_t), detail::kAlign);
    lo.a_lo_high_bytes = lo.a_lo_bytes;
    lo.b_lo_high_bytes = lo.b_lo_bytes;
    lo.a_pk_high_bytes = lo.a_pk_bytes;
    lo.total_bytes  = detail::kAlign; 
    lo.total_bytes += num_moduli * lo.a_lo_bytes;
    lo.total_bytes += num_moduli * lo.b_lo_bytes;
    lo.total_bytes += num_moduli * lo.a_pk_bytes;
    lo.total_bytes += lo.c_hi_bytes;
    lo.total_bytes += num_moduli * lo.c_mid_bytes;
    lo.total_bytes += lo.sft_a_bytes;
    lo.total_bytes += lo.sft_b_bytes;
    lo.total_bytes += lo.s_b_bytes;
    lo.total_bytes += lo.a_lo_high_bytes;
    lo.total_bytes += lo.b_lo_high_bytes;
    lo.total_bytes += lo.a_pk_high_bytes;
    return lo;
}

struct Workspace {
    Layout                     lo;
    std::int8_t               *A_lo; 
    std::int8_t               *B_lo;
    std::int8_t               *A_pk;
    std::int32_t              *C_hi;
    std::vector<std::int8_t *> C_mid_ptrs;
    std::int16_t              *sftA;
    std::int16_t              *sftB;
    std::int32_t              *S_B;
    std::int8_t               *A_lo_high;
    std::int8_t               *B_lo_high;
    std::int8_t               *A_pk_high;
};

inline Workspace slice(void *work, const Layout &lo, unsigned num_moduli) {
    Workspace w{};
    w.lo              = lo;
    auto *p           = reinterpret_cast<std::int8_t *>(detail::alignPtr(work));
    w.A_lo            = p;
    p                += num_moduli * lo.a_lo_bytes;
    w.B_lo            = p;
    p                += num_moduli * lo.b_lo_bytes;
    w.A_pk            = p;
    p                += num_moduli * lo.a_pk_bytes;
    w.C_hi            = reinterpret_cast<std::int32_t *>(p);
    p                += lo.c_hi_bytes;
    w.C_mid_ptrs.resize(num_moduli);
    for (unsigned j = 0; j < num_moduli; ++j) {
        w.C_mid_ptrs[j] = p;
        p              += lo.c_mid_bytes;
    }
    w.sftA = reinterpret_cast<std::int16_t *>(p);
    p     += lo.sft_a_bytes;
    w.sftB = reinterpret_cast<std::int16_t *>(p);
    p     += lo.sft_b_bytes;
    w.S_B = reinterpret_cast<std::int32_t *>(p);
    p    += lo.s_b_bytes;
    w.A_lo_high = p;
    p          += lo.a_lo_high_bytes;
    w.B_lo_high = p;
    p          += lo.b_lo_high_bytes;
    w.A_pk_high = p;
    return w;
}

inline bool is_transposed(Op op) { return op == Op::T; }

template <typename T, unsigned NM>
inline void scale_dispatch(bool cols_major, const T *X, std::size_t ldx,
                           std::size_t k, std::size_t n_outer,
                           std::int8_t *X_lo, std::size_t k_pad,
                           std::size_t inc_lo, std::int16_t *sft_out);

template <>
inline void scale_dispatch<double, 2>(bool cm, const double *X, std::size_t ldx, std::size_t k, std::size_t n_outer,
                                      std::int8_t *X_lo, std::size_t k_pad, std::size_t inc_lo, std::int16_t *s) {
    scaling::scale_double<2>(cm, X, ldx, k, n_outer, X_lo, k_pad, inc_lo, s);
}
#define DISPATCH_SCALE_DOUBLE(NM)                                                                                      \
    template <>                                                                                                        \
    inline void scale_dispatch<double, NM>(bool cm, const double *X, std::size_t ldx, std::size_t k, std::size_t no,    \
                                           std::int8_t *X_lo, std::size_t k_pad, std::size_t inc_lo, std::int16_t *s) {\
        scaling::scale_double<NM>(cm, X, ldx, k, no, X_lo, k_pad, inc_lo, s);                                           \
    }
DISPATCH_SCALE_DOUBLE(3)
DISPATCH_SCALE_DOUBLE(4)
DISPATCH_SCALE_DOUBLE(5)
DISPATCH_SCALE_DOUBLE(6)
DISPATCH_SCALE_DOUBLE(7)
DISPATCH_SCALE_DOUBLE(8)
DISPATCH_SCALE_DOUBLE(9)
DISPATCH_SCALE_DOUBLE(10)
DISPATCH_SCALE_DOUBLE(11)
DISPATCH_SCALE_DOUBLE(12)
DISPATCH_SCALE_DOUBLE(13)
DISPATCH_SCALE_DOUBLE(14)
DISPATCH_SCALE_DOUBLE(15)
DISPATCH_SCALE_DOUBLE(16)
DISPATCH_SCALE_DOUBLE(17)
DISPATCH_SCALE_DOUBLE(18)
DISPATCH_SCALE_DOUBLE(19)
DISPATCH_SCALE_DOUBLE(20)
#undef DISPATCH_SCALE_DOUBLE

#define DISPATCH_SCALE_FLOAT(NM)                                                                                       \
    template <>                                                                                                        \
    inline void scale_dispatch<float, NM>(bool cm, const float *X, std::size_t ldx, std::size_t k, std::size_t no,     \
                                          std::int8_t *X_lo, std::size_t k_pad, std::size_t inc_lo, std::int16_t *s) { \
        scaling::scale_float<NM>(cm, X, ldx, k, no, X_lo, k_pad, inc_lo, s);                                           \
    }
DISPATCH_SCALE_FLOAT(2)
DISPATCH_SCALE_FLOAT(3)
DISPATCH_SCALE_FLOAT(4)
DISPATCH_SCALE_FLOAT(5)
DISPATCH_SCALE_FLOAT(6)
DISPATCH_SCALE_FLOAT(7)
#undef DISPATCH_SCALE_FLOAT

template <unsigned num_moduli>
inline void refine_sft_int8(std::int16_t    *sft_inplace,
                            const std::int32_t *amax_array,
                            std::size_t      len) {
    const float log2P = table::getLog2P_f(num_moduli);
    for (std::size_t i = 0; i < len; ++i) {
        int delta = detail::compute_sft_accu(amax_array[i], log2P);
        int old   = (int)sft_inplace[i];
        sft_inplace[i] = (std::int16_t)(-(old + delta));
    }
}

template <typename T>
inline void run_scale_all(Op           op_A,
                          Op           op_B,
                          std::size_t  m,
                          std::size_t  n,
                          std::size_t  k,
                          const T     *A,
                          std::size_t  lda,
                          const T     *B,
                          std::size_t  ldb,
                          unsigned     num_moduli,
                          const Layout &lo,
                          Workspace   &w);

template <typename T>
inline void run_scale_all_accurate(Op           op_A,
                                   Op           op_B,
                                   std::size_t  m,
                                   std::size_t  n,
                                   std::size_t  k,
                                   const T     *A,
                                   std::size_t  lda,
                                   const T     *B,
                                   std::size_t  ldb,
                                   unsigned     num_moduli,
                                   const Layout &lo,
                                   Workspace   &w);

template <>
inline void run_scale_all<double>(Op op_A, Op op_B,
                                  std::size_t m, std::size_t n, std::size_t k,
                                  const double *A, std::size_t lda,
                                  const double *B, std::size_t ldb,
                                  unsigned num_moduli, const Layout &lo, Workspace &w) {
    bool A_cols_major = is_transposed(op_A);
    std::size_t A_n_outer = m;
    bool B_cols_major = !is_transposed(op_B);
    std::size_t B_n_outer = n;

    const std::size_t inc_A = lo.a_lo_bytes;
    const std::size_t inc_B = lo.b_lo_bytes;

    switch (num_moduli) {
        case 2:  scale_dispatch<double, 2>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 3:  scale_dispatch<double, 3>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 4:  scale_dispatch<double, 4>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 5:  scale_dispatch<double, 5>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 6:  scale_dispatch<double, 6>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 7:  scale_dispatch<double, 7>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 8:  scale_dispatch<double, 8>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 9:  scale_dispatch<double, 9>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 10: scale_dispatch<double, 10>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 11: scale_dispatch<double, 11>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 12: scale_dispatch<double, 12>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 13: scale_dispatch<double, 13>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 14: scale_dispatch<double, 14>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 15: scale_dispatch<double, 15>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 16: scale_dispatch<double, 16>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 17: scale_dispatch<double, 17>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 18: scale_dispatch<double, 18>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 19: scale_dispatch<double, 19>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 20: scale_dispatch<double, 20>(A_cols_major, A, lda, k, A_n_outer, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        default: throw std::invalid_argument("gemmul8_avx512: num_moduli out of supported range [2, 20] for double");
    }
    switch (num_moduli) {
        case 2:  scale_dispatch<double, 2>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 3:  scale_dispatch<double, 3>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 4:  scale_dispatch<double, 4>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 5:  scale_dispatch<double, 5>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 6:  scale_dispatch<double, 6>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 7:  scale_dispatch<double, 7>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 8:  scale_dispatch<double, 8>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 9:  scale_dispatch<double, 9>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 10: scale_dispatch<double, 10>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 11: scale_dispatch<double, 11>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 12: scale_dispatch<double, 12>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 13: scale_dispatch<double, 13>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 14: scale_dispatch<double, 14>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 15: scale_dispatch<double, 15>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 16: scale_dispatch<double, 16>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 17: scale_dispatch<double, 17>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 18: scale_dispatch<double, 18>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 19: scale_dispatch<double, 19>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 20: scale_dispatch<double, 20>(B_cols_major, B, ldb, k, B_n_outer, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        default: break;
    }
}

template <typename T, unsigned NM>
inline void scale_given_sft_dispatch(bool cm, const T *X, std::size_t ldx, std::size_t k, std::size_t n_outer,
                                     std::int8_t *X_lo, std::size_t k_pad, std::size_t inc_lo,
                                     const std::int16_t *sft_in);

template <>
inline void scale_given_sft_dispatch<double, 2>(bool cm, const double *X, std::size_t ldx, std::size_t k, std::size_t n_outer,
                                                std::int8_t *X_lo, std::size_t k_pad, std::size_t inc_lo, const std::int16_t *s) {
    scaling::scale_double_given_sft<2>(cm, X, ldx, k, n_outer, X_lo, k_pad, inc_lo, s);
}
#define DISPATCH_SCALE_GIVEN_D(NM)                                                                                    \
    template <>                                                                                                       \
    inline void scale_given_sft_dispatch<double, NM>(bool cm, const double *X, std::size_t ldx, std::size_t k,         \
                                                     std::size_t no, std::int8_t *X_lo, std::size_t k_pad,             \
                                                     std::size_t inc_lo, const std::int16_t *s) {                     \
        scaling::scale_double_given_sft<NM>(cm, X, ldx, k, no, X_lo, k_pad, inc_lo, s);                               \
    }
DISPATCH_SCALE_GIVEN_D(3)
DISPATCH_SCALE_GIVEN_D(4)
DISPATCH_SCALE_GIVEN_D(5)
DISPATCH_SCALE_GIVEN_D(6)
DISPATCH_SCALE_GIVEN_D(7)
DISPATCH_SCALE_GIVEN_D(8)
DISPATCH_SCALE_GIVEN_D(9)
DISPATCH_SCALE_GIVEN_D(10)
DISPATCH_SCALE_GIVEN_D(11)
DISPATCH_SCALE_GIVEN_D(12)
DISPATCH_SCALE_GIVEN_D(13)
DISPATCH_SCALE_GIVEN_D(14)
DISPATCH_SCALE_GIVEN_D(15)
DISPATCH_SCALE_GIVEN_D(16)
DISPATCH_SCALE_GIVEN_D(17)
DISPATCH_SCALE_GIVEN_D(18)
DISPATCH_SCALE_GIVEN_D(19)
DISPATCH_SCALE_GIVEN_D(20)
#undef DISPATCH_SCALE_GIVEN_D

#define DISPATCH_SCALE_GIVEN_F(NM)                                                                                    \
    template <>                                                                                                       \
    inline void scale_given_sft_dispatch<float, NM>(bool cm, const float *X, std::size_t ldx, std::size_t k,           \
                                                    std::size_t no, std::int8_t *X_lo, std::size_t k_pad,              \
                                                    std::size_t inc_lo, const std::int16_t *s) {                      \
        scaling::scale_float_given_sft<NM>(cm, X, ldx, k, no, X_lo, k_pad, inc_lo, s);                                \
    }
DISPATCH_SCALE_GIVEN_F(2)
DISPATCH_SCALE_GIVEN_F(3)
DISPATCH_SCALE_GIVEN_F(4)
DISPATCH_SCALE_GIVEN_F(5)
DISPATCH_SCALE_GIVEN_F(6)
DISPATCH_SCALE_GIVEN_F(7)
#undef DISPATCH_SCALE_GIVEN_F

template <typename T, unsigned NM>
inline void scale_given_run(Op op_A, Op op_B, std::size_t m, std::size_t n, std::size_t k,
                            const T *A, std::size_t lda, const T *B, std::size_t ldb,
                            const Layout &lo, Workspace &w) {
    bool A_cols_major = is_transposed(op_A);
    bool B_cols_major = !is_transposed(op_B);
    scale_given_sft_dispatch<T, NM>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, lo.a_lo_bytes, w.sftA);
    scale_given_sft_dispatch<T, NM>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, lo.b_lo_bytes, w.sftB);
}

template <typename T>
inline void scale_given_switch(unsigned num_moduli, Op op_A, Op op_B,
                               std::size_t m, std::size_t n, std::size_t k,
                               const T *A, std::size_t lda, const T *B, std::size_t ldb,
                               const Layout &lo, Workspace &w) {
    switch (num_moduli) {
#define CASE_NM(NM)                                                                                                  \
    case NM:                                                                                                          \
        scale_given_run<T, NM>(op_A, op_B, m, n, k, A, lda, B, ldb, lo, w);                                           \
        break
    CASE_NM(2); CASE_NM(3); CASE_NM(4); CASE_NM(5); CASE_NM(6); CASE_NM(7);
    default:
        if constexpr (std::is_same_v<T, double>) {
            switch (num_moduli) {
                CASE_NM(8); CASE_NM(9); CASE_NM(10); CASE_NM(11); CASE_NM(12); CASE_NM(13);
                CASE_NM(14); CASE_NM(15); CASE_NM(16); CASE_NM(17); CASE_NM(18); CASE_NM(19); CASE_NM(20);
                default: throw std::invalid_argument("num_moduli out of range");
            }
        } else {
            throw std::invalid_argument("num_moduli out of range for float");
        }
#undef CASE_NM
    }
}

template <>
inline void run_scale_all_accurate<double>(Op op_A, Op op_B,
                                           std::size_t m, std::size_t n, std::size_t k,
                                           const double *A, std::size_t lda,
                                           const double *B, std::size_t ldb,
                                           unsigned num_moduli, const Layout &lo, Workspace &w) {
    bool A_cols_major = is_transposed(op_A);
    bool B_cols_major = !is_transposed(op_B);

    scaling::extract_high_order_double<double>(A_cols_major, A, lda, k, m,
                                               w.A_lo_high, lo.k_pad, w.sftA);
    scaling::extract_high_order_double<double>(B_cols_major, B, ldb, k, n,
                                               w.B_lo_high, lo.k_pad, w.sftB);

    matmult::compute_S_B(w.B_lo_high, lo.k_pad, n, lo.n_pad, w.S_B);
    std::vector<std::int32_t> S_B_neg128(lo.n_pad);
    for (std::size_t i = 0; i < lo.n_pad; ++i) S_B_neg128[i] = -128 * w.S_B[i];
    matmult::pack_A(w.A_lo_high, lo.k_pad, m, lo.m_pad, w.A_pk_high, true);
    matmult::gemm_i8x1(w.A_pk_high, w.B_lo_high, w.C_hi, lo.m_pad,
                       m, n, lo.k_pad, lo.m_pad, lo.n_pad, S_B_neg128.data());

    std::vector<std::int32_t> amax_rows(m), amax_cols(n);
    matmult::c_hi_row_absmax(w.C_hi, m, n, lo.m_pad, amax_rows.data());
    matmult::c_hi_col_absmax(w.C_hi, m, n, lo.m_pad, amax_cols.data());

    switch (num_moduli) {
#define REFINE_CASE(NM)                                                             \
    case NM:                                                                         \
        refine_sft_int8<NM>(w.sftA, amax_rows.data(), m);                            \
        refine_sft_int8<NM>(w.sftB, amax_cols.data(), n);                            \
        break
    REFINE_CASE(2); REFINE_CASE(3); REFINE_CASE(4); REFINE_CASE(5);
    REFINE_CASE(6); REFINE_CASE(7); REFINE_CASE(8); REFINE_CASE(9);
    REFINE_CASE(10); REFINE_CASE(11); REFINE_CASE(12); REFINE_CASE(13);
    REFINE_CASE(14); REFINE_CASE(15); REFINE_CASE(16); REFINE_CASE(17);
    REFINE_CASE(18); REFINE_CASE(19); REFINE_CASE(20);
#undef REFINE_CASE
    default:
        throw std::invalid_argument("accurate mode: num_moduli out of range [2, 20]");
    }

    scale_given_switch<double>(num_moduli, op_A, op_B, m, n, k, A, lda, B, ldb, lo, w);
}

template <>
inline void run_scale_all_accurate<float>(Op op_A, Op op_B,
                                          std::size_t m, std::size_t n, std::size_t k,
                                          const float *A, std::size_t lda,
                                          const float *B, std::size_t ldb,
                                          unsigned num_moduli, const Layout &lo, Workspace &w) {
    bool A_cols_major = is_transposed(op_A);
    bool B_cols_major = !is_transposed(op_B);
    scaling::extract_high_order_double<float>(A_cols_major, A, lda, k, m,
                                              w.A_lo_high, lo.k_pad, w.sftA);
    scaling::extract_high_order_double<float>(B_cols_major, B, ldb, k, n,
                                              w.B_lo_high, lo.k_pad, w.sftB);
    matmult::compute_S_B(w.B_lo_high, lo.k_pad, n, lo.n_pad, w.S_B);
    std::vector<std::int32_t> S_B_neg128(lo.n_pad);
    for (std::size_t i = 0; i < lo.n_pad; ++i) S_B_neg128[i] = -128 * w.S_B[i];
    matmult::pack_A(w.A_lo_high, lo.k_pad, m, lo.m_pad, w.A_pk_high, true);
    matmult::gemm_i8x1(w.A_pk_high, w.B_lo_high, w.C_hi, lo.m_pad,
                       m, n, lo.k_pad, lo.m_pad, lo.n_pad, S_B_neg128.data());

    std::vector<std::int32_t> amax_rows(m), amax_cols(n);
    matmult::c_hi_row_absmax(w.C_hi, m, n, lo.m_pad, amax_rows.data());
    matmult::c_hi_col_absmax(w.C_hi, m, n, lo.m_pad, amax_cols.data());

    switch (num_moduli) {
#define REFINE_CASE_F(NM)                                                            \
    case NM:                                                                          \
        refine_sft_int8<NM>(w.sftA, amax_rows.data(), m);                             \
        refine_sft_int8<NM>(w.sftB, amax_cols.data(), n);                             \
        break
    REFINE_CASE_F(2); REFINE_CASE_F(3); REFINE_CASE_F(4); REFINE_CASE_F(5);
    REFINE_CASE_F(6); REFINE_CASE_F(7);
#undef REFINE_CASE_F
    default:
        throw std::invalid_argument("accurate mode (float): num_moduli out of range [2, 7]");
    }

    scale_given_switch<float>(num_moduli, op_A, op_B, m, n, k, A, lda, B, ldb, lo, w);
}

template <>
inline void run_scale_all<float>(Op op_A, Op op_B,
                                 std::size_t m, std::size_t n, std::size_t k,
                                 const float *A, std::size_t lda,
                                 const float *B, std::size_t ldb,
                                 unsigned num_moduli, const Layout &lo, Workspace &w) {
    bool A_cols_major = is_transposed(op_A);
    bool B_cols_major = !is_transposed(op_B);
    const std::size_t inc_A = lo.a_lo_bytes;
    const std::size_t inc_B = lo.b_lo_bytes;
    switch (num_moduli) {
        case 2: scale_dispatch<float, 2>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 3: scale_dispatch<float, 3>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 4: scale_dispatch<float, 4>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 5: scale_dispatch<float, 5>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 6: scale_dispatch<float, 6>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        case 7: scale_dispatch<float, 7>(A_cols_major, A, lda, k, m, w.A_lo, lo.k_pad, inc_A, w.sftA); break;
        default: throw std::invalid_argument("gemmul8_avx512: num_moduli out of supported range [2, 7] for float");
    }
    switch (num_moduli) {
        case 2: scale_dispatch<float, 2>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 3: scale_dispatch<float, 3>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 4: scale_dispatch<float, 4>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 5: scale_dispatch<float, 5>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 6: scale_dispatch<float, 6>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        case 7: scale_dispatch<float, 7>(B_cols_major, B, ldb, k, n, w.B_lo, lo.k_pad, inc_B, w.sftB); break;
        default: break;
    }
}

template <typename T>
std::vector<double> gemm_impl(Op op_A, Op op_B,
                              std::size_t m, std::size_t n, std::size_t k,
                              T alpha, const T *A, std::size_t lda,
                              const T *B, std::size_t ldb,
                              T beta, T *C, std::size_t ldc,
                              unsigned num_moduli, void *work, bool fastmode) {
    if (num_moduli < 2) throw std::invalid_argument("gemmul8_avx512: num_moduli must be >= 2");
    const Layout lo = compute_layout(m, n, k, num_moduli);
    Workspace    w  = slice(work, lo, num_moduli);

    std::vector<double> timer(4, 0.0);
    auto tic = []() { return std::chrono::steady_clock::now(); };
    auto toc = [](auto &ts, double &acc) {
        auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration_cast<std::chrono::nanoseconds>(now - ts).count();
        ts   = now;
    };

    auto t0 = tic();
    if (fastmode) {
        run_scale_all<T>(op_A, op_B, m, n, k, A, lda, B, ldb, num_moduli, lo, w);
    } else {
        run_scale_all_accurate<T>(op_A, op_B, m, n, k, A, lda, B, ldb, num_moduli, lo, w);
    }
    for (unsigned j = 0; j < num_moduli; ++j) {
        matmult::pack_A(w.A_lo + j * lo.a_lo_bytes, lo.k_pad, m, lo.m_pad,
                        w.A_pk + j * lo.a_pk_bytes, true);
    }
    toc(t0, timer[0]);

    auto t1 = tic();
    std::vector<std::int32_t> S_B_neg128(lo.n_pad);
    for (unsigned j = 0; j < num_moduli; ++j) {
        matmult::compute_S_B(w.B_lo + j * lo.b_lo_bytes, lo.k_pad, n, lo.n_pad, w.S_B);
        for (std::size_t nn = 0; nn < lo.n_pad; ++nn) S_B_neg128[nn] = -128 * w.S_B[nn];

        matmult::gemm_i8x1(w.A_pk + j * lo.a_pk_bytes,
                           w.B_lo + j * lo.b_lo_bytes,
                           w.C_hi, lo.m_pad, m, n, lo.k_pad, lo.m_pad, lo.n_pad,
                           S_B_neg128.data());
        toc(t1, timer[1]);

        std::int8_t *C_mid_j = w.C_mid_ptrs[j];
#pragma omp parallel for schedule(static)
        for (std::size_t nn = 0; nn < n; ++nn) {
            mod_pass::conv_hi2mid(j,
                                  w.C_hi + nn * lo.m_pad,
                                  m,
                                  C_mid_j + nn * m);
        }
        toc(t1, timer[2]);
    }

    auto t2 = tic();
    switch (num_moduli) {
#define CASE_NM(NM)                                                                                              \
    case NM:                                                                                                     \
        invscal::inverse_scale_and_writeback<T, NM>(                                                             \
            const_cast<const std::int8_t *const *>(w.C_mid_ptrs.data()), m, m, n, alpha, beta, C, ldc,           \
            w.sftA, w.sftB);                                                                                     \
        break
        CASE_NM(2);
        CASE_NM(3);
        CASE_NM(4);
        CASE_NM(5);
        CASE_NM(6);
        CASE_NM(7);
        CASE_NM(8);
        CASE_NM(9);
        CASE_NM(10);
        CASE_NM(11);
        CASE_NM(12);
        CASE_NM(13);
        CASE_NM(14);
        CASE_NM(15);
        CASE_NM(16);
        CASE_NM(17);
        CASE_NM(18);
        CASE_NM(19);
        CASE_NM(20);
#undef CASE_NM
        default: break;
    }
    toc(t2, timer[3]);
    return timer;
}

} // namespace

template <typename T>
std::size_t workSize(std::size_t m, std::size_t n, std::size_t k, unsigned num_moduli,
                     std::size_t *wsA, std::size_t *wsB) {
    Layout lo = compute_layout(m, n, k, num_moduli);
    if (wsA) *wsA = num_moduli * lo.a_lo_bytes;
    if (wsB) *wsB = num_moduli * lo.b_lo_bytes;
    return lo.total_bytes;
}

template std::size_t workSize<double>(std::size_t, std::size_t, std::size_t, unsigned, std::size_t *, std::size_t *);
template std::size_t workSize<float>(std::size_t, std::size_t, std::size_t, unsigned, std::size_t *, std::size_t *);

template <typename T>
std::vector<double> gemm(Op op_A, Op op_B, std::size_t m, std::size_t n, std::size_t k,
                         T alpha, const T *A, std::size_t lda, const T *B, std::size_t ldb,
                         T beta, T *C, std::size_t ldc, unsigned num_moduli, void *work,
                         bool fastmode) {
    return gemm_impl<T>(op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, work, fastmode);
}

template std::vector<double> gemm<double>(Op, Op, std::size_t, std::size_t, std::size_t, double,
                                          const double *, std::size_t, const double *, std::size_t,
                                          double, double *, std::size_t, unsigned, void *, bool);
template std::vector<double> gemm<float>(Op, Op, std::size_t, std::size_t, std::size_t, float,
                                         const float *, std::size_t, const float *, std::size_t,
                                         float, float *, std::size_t, unsigned, void *, bool);

} // namespace gemmul8_avx512
