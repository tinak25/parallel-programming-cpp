// POSIX threads: the rows of C are split into one contiguous block per thread.
// Each thread writes only its own rows, so no locking is needed.
#include <pthread.h>

#include <vector>

#include "matrix.hpp"

namespace {
struct Job {
    const int* a;
    const int* b;
    int* c;
    int n;
    int row_begin, row_end;
    const std::string* order;
};

void* worker(void* p) {
    const Job* j = static_cast<const Job*>(p);
    mm::multiply_rows(*j->order, j->a, j->b, j->c, j->n, j->row_begin, j->row_end);
    return nullptr;
}
}  // namespace

int main(int argc, char** argv) {
    using namespace mm;
    const Args a = parse_args(argc, argv);
    const Matrix A = random_matrix(a.n, a.seed), B = random_matrix(a.n, a.seed + 1);
    Matrix C(A.size());

    std::vector<pthread_t> threads(a.threads);
    std::vector<Job> jobs(a.threads);
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < a.threads; ++t) {
        // Spread the remainder: the first n % threads blocks get one extra row.
        const int base = a.n / a.threads, extra = a.n % a.threads;
        const int begin = t * base + (t < extra ? t : extra);
        const int end = begin + base + (t < extra ? 1 : 0);
        jobs[t] = Job{A.data(), B.data(), C.data(), a.n, begin, end, &a.order};
        if (pthread_create(&threads[t], nullptr, worker, &jobs[t]) != 0) {
            std::cerr << "pthread_create failed\n";
            return 1;
        }
    }
    for (auto& th : threads) pthread_join(th, nullptr);
    const double secs = seconds_since(t0);

    return report("pthreads-" + a.order + "-t" + std::to_string(a.threads), a, C, secs, &A, &B);
}
