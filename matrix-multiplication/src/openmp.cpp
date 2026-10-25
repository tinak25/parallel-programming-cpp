// OpenMP: one pragma splits the row loop across threads.
//
// Must be compiled with -fopenmp (on macOS with Apple clang:
// -Xpreprocessor -fopenmp plus libomp). Without the flag the pragma is
// silently ignored and the program runs on one thread, so this program
// refuses to run in that case instead of reporting misleading timings.
#ifndef _OPENMP
#error "Compile with -fopenmp, otherwise #pragma omp is silently ignored"
#endif
#include <omp.h>

#include "matrix.hpp"

int main(int argc, char** argv) {
    using namespace mm;
    const Args a = parse_args(argc, argv);
    omp_set_num_threads(a.threads);
    const Matrix A = random_matrix(a.n, a.seed), B = random_matrix(a.n, a.seed + 1);
    Matrix C(A.size());
    const int n = a.n;

    const auto t0 = std::chrono::steady_clock::now();
    if (a.order == "ijk") {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) multiply_rows_ijk(A.data(), B.data(), C.data(), n, i, i + 1);
    } else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) multiply_rows_ikj(A.data(), B.data(), C.data(), n, i, i + 1);
    }
    const double secs = seconds_since(t0);

    return report("openmp-" + a.order + "-t" + std::to_string(a.threads), a, C, secs, &A, &B);
}
