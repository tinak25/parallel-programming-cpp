// Quicksort variants, a thread-parallel quicksort, a k-way merge, and the
// command-line handling shared by the sorting programs.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace srt {

inline std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// n integers in [0, range). range = 1000 reproduces the original assignment
// data (lots of repeated values); a large range gives mostly distinct values.
inline std::vector<int> make_data(long long n, long long range, std::uint64_t seed) {
    std::vector<int> v(static_cast<std::size_t>(n));
    for (auto& x : v) x = static_cast<int>(splitmix64(seed) % static_cast<std::uint64_t>(range));
    return v;
}

constexpr int kInsertionCutoff = 16;

inline void insertion_sort(int* a, long long lo, long long hi) {  // inclusive bounds
    for (long long i = lo + 1; i <= hi; ++i) {
        const int x = a[i];
        long long j = i - 1;
        while (j >= lo && a[j] > x) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = x;
    }
}

inline int median_of_three(int* a, long long lo, long long hi) {
    const long long mid = lo + (hi - lo) / 2;
    int x = a[lo], y = a[mid], z = a[hi];
    if (x > y) std::swap(x, y);
    if (y > z) std::swap(y, z);
    if (x > y) std::swap(x, y);
    return y;
}

// Lomuto partition, as in the original coursework: everything smaller than the
// pivot goes left, everything else (including values EQUAL to the pivot) goes
// right. With many repeated values, a block of equal values only shrinks by one
// element per partition, so the sort becomes quadratic.
inline void quicksort_lomuto(int* a, long long lo, long long hi) {
    while (hi - lo > kInsertionCutoff) {
        std::swap(a[hi], a[lo + (hi - lo) / 2]);  // middle element as pivot, moved to the end
        const int pivot = a[hi];
        long long i = lo - 1;
        for (long long j = lo; j < hi; ++j)
            if (a[j] < pivot) std::swap(a[++i], a[j]);
        std::swap(a[i + 1], a[hi]);
        const long long p = i + 1;
        // Recurse into the smaller side, loop on the larger: stack depth stays O(log n).
        if (p - lo < hi - p) {
            quicksort_lomuto(a, lo, p - 1);
            lo = p + 1;
        } else {
            quicksort_lomuto(a, p + 1, hi);
            hi = p - 1;
        }
    }
    insertion_sort(a, lo, hi);
}

// Three-way (Dijkstra) partition: <, == and > the pivot. All copies of the
// pivot are finished in one pass, so repeated values make the sort FASTER.
inline void partition3(int* a, long long lo, long long hi, long long& lt, long long& gt) {
    const int pivot = median_of_three(a, lo, hi);
    lt = lo;
    gt = hi;
    long long i = lo;
    while (i <= gt) {
        if (a[i] < pivot) std::swap(a[lt++], a[i++]);
        else if (a[i] > pivot) std::swap(a[i], a[gt--]);
        else ++i;
    }
}

inline void quicksort_3way(int* a, long long lo, long long hi) {
    while (hi - lo > kInsertionCutoff) {
        long long lt, gt;
        partition3(a, lo, hi, lt, gt);
        if (lt - lo < hi - gt) {
            quicksort_3way(a, lo, lt - 1);
            lo = gt + 1;
        } else {
            quicksort_3way(a, gt + 1, hi);
            hi = lt - 1;
        }
    }
    insertion_sort(a, lo, hi);
}

inline void sort_with(const std::string& algo, std::vector<int>& v) {
    if (v.empty()) return;
    const long long hi = static_cast<long long>(v.size()) - 1;
    if (algo == "lomuto") quicksort_lomuto(v.data(), 0, hi);
    else if (algo == "3way") quicksort_3way(v.data(), 0, hi);
    else std::sort(v.begin(), v.end());
}

// Thread-parallel 3-way quicksort. After partitioning, the left part is handed
// to a new thread while the current thread continues with the right part.
// New threads are only created while `depth` > 0 and the part is big enough
// to be worth a thread; below that, the sort continues sequentially.
inline void parallel_quicksort(int* a, long long lo, long long hi, int depth) {
    constexpr long long kMinParallel = 50000;
    if (depth <= 0 || hi - lo < kMinParallel) {
        if (hi > lo) quicksort_3way(a, lo, hi);
        return;
    }
    long long lt, gt;
    partition3(a, lo, hi, lt, gt);
    std::thread left([=] { parallel_quicksort(a, lo, lt - 1, depth - 1); });
    parallel_quicksort(a, gt + 1, hi, depth - 1);
    left.join();
}

// Spawns enough levels that there are roughly 4 parts per thread, which
// smooths out uneven pivots.
inline int depth_for_threads(int threads) {
    int depth = 0;
    while ((1 << depth) < threads * 4) ++depth;
    return threads <= 1 ? 0 : depth;
}

// Merges `parts` sorted runs (given by start offsets and lengths) with a
// min-heap: O(N log k). The original coursework scanned all k runs for every
// output element, O(N k), and in the OpenCL version re-sorted the whole array
// with std::sort, which also hid whether the parallel step had worked.
inline std::vector<int> kway_merge(const std::vector<int>& data, const std::vector<int>& offsets,
                                   const std::vector<int>& counts) {
    using Item = std::pair<int, std::size_t>;  // (value, run index)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    std::vector<int> pos(offsets.size(), 0);
    for (std::size_t r = 0; r < offsets.size(); ++r)
        if (counts[r] > 0) heap.push({data[offsets[r]], r});
    std::vector<int> out;
    out.reserve(data.size());
    while (!heap.empty()) {
        auto [value, r] = heap.top();
        heap.pop();
        out.push_back(value);
        if (++pos[r] < counts[r]) heap.push({data[offsets[r] + pos[r]], r});
    }
    return out;
}

inline std::uint64_t checksum(const std::vector<int>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (int x : v) {
        h ^= static_cast<std::uint32_t>(x);
        h *= 1099511628211ULL;
    }
    return h ^ v.size();
}

struct Args {
    long long n = 1000000;
    long long range = 1000;
    std::string algo = "3way";      // lomuto | 3way | std
    std::string kernel = "bitonic"; // OpenCL: oddeven | bitonic
    int threads = 2;
    std::uint64_t seed = 11;
    bool verify = false;
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << s << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (s == "--n") a.n = std::stoll(next());
        else if (s == "--range") a.range = std::stoll(next());
        else if (s == "--algo") a.algo = next();
        else if (s == "--kernel") a.kernel = next();
        else if (s == "--threads") a.threads = std::stoi(next());
        else if (s == "--seed") a.seed = std::stoull(next());
        else if (s == "--verify") a.verify = true;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--n N] [--range R] [--algo lomuto|3way|std] [--threads T]\n"
                         "       [--kernel oddeven|bitonic] [--seed S] [--verify]\n";
            std::exit(s == "--help" ? 0 : 1);
        }
    }
    if (a.n < 0 || a.range < 1 || a.range > 2147483647LL || a.threads < 1 ||
        (a.algo != "lomuto" && a.algo != "3way" && a.algo != "std") ||
        (a.kernel != "oddeven" && a.kernel != "bitonic")) {
        std::cerr << "invalid arguments\n";
        std::exit(1);
    }
    return a;
}

// Prints the lines the test and benchmark scripts parse. With --verify the
// output is compared with std::sort on the same input: same length, same
// values, same order.
inline int report(const std::string& version, const Args& a, const std::vector<int>& input,
                  const std::vector<int>& sorted, double seconds) {
    std::cout << "version: " << version << "\nn: " << sorted.size() << "\nchecksum: " << checksum(sorted)
              << "\nseconds: " << seconds << "\n";
    if (a.verify) {
        std::vector<int> expected = input;
        std::sort(expected.begin(), expected.end());
        const bool ok = expected == sorted;
        std::cout << "verify: " << (ok ? "ok" : "MISMATCH") << "\n";
        return ok ? 0 : 2;
    }
    return 0;
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace srt
