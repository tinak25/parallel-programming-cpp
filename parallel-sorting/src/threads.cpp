// Thread-parallel 3-way quicksort (std::thread). Each partition step hands
// one side to a new thread, down to a fixed depth.
#include "sorting.hpp"

int main(int argc, char** argv) {
    using namespace srt;
    const Args a = parse_args(argc, argv);
    const std::vector<int> input = make_data(a.n, a.range, a.seed);
    std::vector<int> v = input;

    const auto t0 = std::chrono::steady_clock::now();
    if (!v.empty()) parallel_quicksort(v.data(), 0, static_cast<long long>(v.size()) - 1, depth_for_threads(a.threads));
    const double secs = seconds_since(t0);

    return report("threads-t" + std::to_string(a.threads), a, input, v, secs);
}
