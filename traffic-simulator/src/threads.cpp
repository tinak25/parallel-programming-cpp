// Producer-consumer with any number of producers and consumers.
//
// Producers play the role of sensors: each one owns a contiguous slice of the
// readings and pushes raw lines into a bounded queue (optionally waiting
// --sensor-delay-us between readings, like a real sensor). Consumers parse the
// lines and aggregate into their own private counts, which are merged once at
// the end, so the hot loop takes no lock except the queue's own.
//
// --batch N sends N lines per queue operation instead of one. Parsing a line
// takes a fraction of a microsecond, less than one push/pop through the
// mutex-protected queue, so without batching the queue itself is the bottleneck.
#include <chrono>
#include <mutex>
#include <thread>

#include "bounded_queue.hpp"
#include "traffic.hpp"

int main(int argc, char** argv) {
    using namespace traffic;
    const Args a = parse_args(argc, argv);
    const std::vector<std::string> lines = read_lines(a.file);

    const auto t0 = std::chrono::steady_clock::now();
    using Range = std::pair<std::size_t, std::size_t>;  // [first, last) line indices
    BoundedQueue<Range> queue(static_cast<std::size_t>(a.queue));
    const std::size_t batch = static_cast<std::size_t>(a.batch);

    std::vector<std::thread> producers;
    const std::size_t total = lines.size();
    for (int p = 0; p < a.producers; ++p) {
        const std::size_t begin = total * p / a.producers, end = total * (p + 1) / a.producers;
        producers.emplace_back([&, begin, end] {
            for (std::size_t i = begin; i < end; i += batch) {
                const std::size_t last = std::min(end, i + batch);
                if (a.sensor_delay_us)  // a sensor produces one reading per delay
                    std::this_thread::sleep_for(std::chrono::microseconds(a.sensor_delay_us * (last - i)));
                queue.push({i, last});
            }
        });
    }

    HourlyCounts counts;
    long readings = 0, skipped = 0;
    std::mutex merge_mutex;
    std::vector<std::thread> consumers;
    for (int c = 0; c < a.consumers; ++c) {
        consumers.emplace_back([&] {
            HourlyCounts local;
            long ok = 0, bad = 0;
            while (auto range = queue.pop()) {
                for (std::size_t i = range->first; i < range->second; ++i) {
                    if (auto r = parse_line(lines[i])) {
                        add(local, *r);
                        ++ok;
                    } else {
                        ++bad;
                    }
                }
            }
            std::lock_guard<std::mutex> lock(merge_mutex);
            merge(counts, local);
            readings += ok;
            skipped += bad;
        });
    }

    for (auto& t : producers) t.join();
    queue.close();  // only after every producer is done
    for (auto& t : consumers) t.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    print_result("threads-p" + std::to_string(a.producers) + "-c" + std::to_string(a.consumers) + "-b" +
                     std::to_string(a.batch),
                 a, counts,
                 readings, skipped, secs);
    return 0;
}
