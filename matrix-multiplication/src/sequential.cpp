// Single-threaded baseline. --order ijk is the textbook loop order,
// --order ikj is the cache-friendly order.
#include "matrix.hpp"

int main(int argc, char** argv) {
    using namespace mm;
    const Args a = parse_args(argc, argv);
    const Matrix A = random_matrix(a.n, a.seed), B = random_matrix(a.n, a.seed + 1);
    Matrix C(A.size());

    const auto t0 = std::chrono::steady_clock::now();  // time the multiplication only
    multiply_rows(a.order, A.data(), B.data(), C.data(), a.n, 0, a.n);
    const double secs = seconds_since(t0);

    return report("sequential-" + a.order, a, C, secs, &A, &B);
}
