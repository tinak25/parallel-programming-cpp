// Baseline: read, parse and aggregate in one thread.
#include <chrono>
#include <thread>

#include "traffic.hpp"

int main(int argc, char** argv) {
    using namespace traffic;
    const Args a = parse_args(argc, argv);
    const std::vector<std::string> lines = read_lines(a.file);

    const auto t0 = std::chrono::steady_clock::now();
    HourlyCounts counts;
    long readings = 0, skipped = 0;
    for (const auto& line : lines) {
        if (a.sensor_delay_us) std::this_thread::sleep_for(std::chrono::microseconds(a.sensor_delay_us));
        if (auto r = parse_line(line)) {
            add(counts, *r);
            ++readings;
        } else {
            ++skipped;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    print_result("sequential", a, counts, readings, skipped, secs);
    return 0;
}
