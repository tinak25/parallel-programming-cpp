// Single-threaded sort. --algo lomuto reproduces the original partition,
// --algo 3way is the fix, --algo std is std::sort for reference.
#include "sorting.hpp"

int main(int argc, char** argv) {
    using namespace srt;
    const Args a = parse_args(argc, argv);
    const std::vector<int> input = make_data(a.n, a.range, a.seed);
    std::vector<int> v = input;

    const auto t0 = std::chrono::steady_clock::now();
    sort_with(a.algo, v);
    const double secs = seconds_since(t0);

    return report("sequential-" + a.algo, a, input, v, secs);
}
