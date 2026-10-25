// N x N integer matrices stored row-major in one flat array, plus the two
// single-threaded multiplication orders and a checksum used to compare the
// output of every version.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace mm {

using Matrix = std::vector<int>;  // element (i, j) is at i * n + j

// Platform-independent random numbers so every version, on every OS,
// multiplies exactly the same matrices.
inline std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline Matrix random_matrix(int n, std::uint64_t seed) {
    Matrix m(static_cast<std::size_t>(n) * n);
    for (auto& x : m) x = static_cast<int>(splitmix64(seed) % 10);  // 0-9, so sums never overflow
    return m;
}

// Textbook order: C[i][j] = sum over k of A[i][k] * B[k][j].
// The inner loop walks DOWN a column of B, jumping n ints each step, so almost
// every read of B misses the CPU cache once n is a few hundred.
inline void multiply_rows_ijk(const int* a, const int* b, int* c, int n, int row_begin, int row_end) {
    for (int i = row_begin; i < row_end; ++i)
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

// Same arithmetic, loops reordered: for each A[i][k], add A[i][k] * (row k of B)
// to row i of C. Every inner-loop access is now sequential in memory, which
// the cache and the compiler's vectoriser both like.
inline void multiply_rows_ikj(const int* a, const int* b, int* c, int n, int row_begin, int row_end) {
    for (int i = row_begin; i < row_end; ++i) {
        int* crow = c + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) crow[j] = 0;
        for (int k = 0; k < n; ++k) {
            const int aik = a[i * n + k];
            const int* brow = b + static_cast<std::size_t>(k) * n;
            for (int j = 0; j < n; ++j) crow[j] += aik * brow[j];
        }
    }
}

inline void multiply_rows(const std::string& order, const int* a, const int* b, int* c, int n, int r0, int r1) {
    if (order == "ijk") multiply_rows_ijk(a, b, c, n, r0, r1);
    else multiply_rows_ikj(a, b, c, n, r0, r1);
}

// Order-sensitive checksum: two results match only if every element matches
// (up to the astronomically unlikely hash collision).
inline std::uint64_t checksum(const Matrix& c) {
    std::uint64_t h = 1469598103934665603ULL;
    for (int x : c) {
        h ^= static_cast<std::uint32_t>(x);
        h *= 1099511628211ULL;
    }
    return h;
}

struct Args {
    int n = 1024;
    int threads = 2;
    std::string order = "ikj";
    std::uint64_t seed = 7;
    bool verify = false;  // recompute single-threaded and compare element by element
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << s << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (s == "--n") a.n = std::stoi(next());
        else if (s == "--threads") a.threads = std::stoi(next());
        else if (s == "--order") a.order = next();
        else if (s == "--seed") a.seed = std::stoull(next());
        else if (s == "--verify") a.verify = true;
        else {
            std::cerr << "usage: " << argv[0] << " [--n N] [--threads T] [--order ijk|ikj] [--seed S] [--verify]\n";
            std::exit(s == "--help" ? 0 : 1);
        }
    }
    if (a.n < 1 || a.threads < 1 || (a.order != "ijk" && a.order != "ikj")) {
        std::cerr << "invalid arguments\n";
        std::exit(1);
    }
    return a;
}

// Single-threaded reference result.
inline Matrix reference(const Matrix& a, const Matrix& b, int n) {
    Matrix c(a.size());
    multiply_rows_ikj(a.data(), b.data(), c.data(), n, 0, n);
    return c;
}

// Every program ends by printing the same three lines, which the test and
// benchmark scripts parse.
inline int report(const std::string& version, const Args& a, const Matrix& c, double seconds,
                  const Matrix* a_in = nullptr, const Matrix* b_in = nullptr) {
    std::cout << "version: " << version << "\n"
              << "checksum: " << checksum(c) << "\n"
              << "seconds: " << seconds << "\n";
    if (a.verify && a_in && b_in) {
        const bool ok = reference(*a_in, *b_in, a.n) == c;
        std::cout << "verify: " << (ok ? "ok" : "MISMATCH") << "\n";
        return ok ? 0 : 2;
    }
    return 0;
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace mm
