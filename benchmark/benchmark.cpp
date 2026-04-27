#include "../avx-512/include/gemmul8_avx512.hpp"
#include "../amx/include/gemmul8_amx.hpp"

#include <mkl.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

namespace {
bool enable_amx() {
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    return rc == 0;
}
} // namespace

namespace {

constexpr std::uint64_t kSeed = 1337ULL;
constexpr std::size_t   kDefaultIters  = 0x10;
constexpr unsigned      kDefaultModuli = 14;

const std::vector<std::size_t> kDefaultSizes = {64, 128, 256, 512, 1024, 2048, 4096};

double gflops(std::size_t m, std::size_t n, std::size_t k, double seconds) {
    if (seconds <= 0.0) return 0.0;
    return 2.0 * (double)m * (double)n * (double)k / seconds * 1e-9;
}

double checksum(const double *C, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += C[i];
    return s;
}

std::vector<std::size_t> parse_sizes(const char *arg) {
    std::vector<std::size_t> out;
    const char *p = arg;
    while (*p) {
        char *end = nullptr;
        std::size_t v = std::strtoull(p, &end, 10);
        if (end == p) break;
        out.push_back(v);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    if (!enable_amx()) {
        std::fprintf(stderr,
            "warning: arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA) failed; "
            "AMX kernel will SIGILL\n");
    }

    std::size_t iters       = kDefaultIters;
    unsigned    num_moduli  = kDefaultModuli;
    std::vector<std::size_t> sizes = kDefaultSizes;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) {
            iters = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--moduli" && i + 1 < argc) {
            num_moduli = (unsigned)std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--sizes" && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
        } else {
            std::fprintf(stderr,
                "usage: %s [--iters N] [--moduli M] [--sizes n1,n2,...]\n",
                argv[0]);
            return 1;
        }
    }

    std::printf("# iters=%zu moduli=%u\n", iters, num_moduli);
    std::printf("# %-8s %-12s %12s %12s %12s\n",
                "size", "backend", "time_s", "gflops", "checksum");

    for (std::size_t m : sizes) {
        const std::size_t n = m, k = m;
        const std::size_t lda = m, ldb = k, ldc = m;

        std::vector<double> A(m * k), B(k * n), C0(m * n);
        std::mt19937_64 rng(kSeed);
        std::uniform_real_distribution<double> du(-1.0, 1.0);
        for (auto &x : A)  x = du(rng);
        for (auto &x : B)  x = du(rng);
        for (auto &x : C0) x = du(rng);

        const double alpha = du(rng);
        std::uniform_real_distribution<double> db(-0.99, 0.99);
        const double beta = db(rng);

        std::size_t wsz_avx = gemmul8_avx512::workSize<double>(m, n, k, num_moduli);
        std::size_t wsz_amx = gemmul8_amx::workSize<double>(m, n, k, num_moduli);
        std::vector<char> work_avx(wsz_avx + 4096);
        std::vector<char> work_amx(wsz_amx + 4096);

        std::vector<double> C(m * n);

        auto run = [&](const char *name, auto &&kernel) {
            std::memcpy(C.data(), C0.data(), C.size() * sizeof(double));
            auto t0 = std::chrono::steady_clock::now();
            for (std::size_t it = 0; it < iters; ++it) kernel();
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            double per_call = secs / (double)iters;
            std::printf("  %-8zu %-12s %12.6f %12.3f %12.6e\n",
                        m, name, secs, gflops(m, n, k, per_call),
                        checksum(C.data(), C.size()));
            std::fflush(stdout);
        };

        run("avx512", [&] {
            gemmul8_avx512::gemm<double>(
                gemmul8_avx512::Op::N, gemmul8_avx512::Op::N,
                m, n, k, alpha, A.data(), lda, B.data(), ldb,
                beta, C.data(), ldc, num_moduli, work_avx.data(),
                /*fastmode=*/false);
        });

        run("amx", [&] {
            gemmul8_amx::gemm<double>(
                gemmul8_amx::Op::N, gemmul8_amx::Op::N,
                m, n, k, alpha, A.data(), lda, B.data(), ldb,
                beta, C.data(), ldc, num_moduli, work_amx.data(),
                /*fastmode=*/false);
        });

        run("mkl", [&] {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        (MKL_INT)m, (MKL_INT)n, (MKL_INT)k,
                        alpha, A.data(), (MKL_INT)lda,
                        B.data(), (MKL_INT)ldb,
                        beta, C.data(), (MKL_INT)ldc);
        });
    }

    return 0;
}
