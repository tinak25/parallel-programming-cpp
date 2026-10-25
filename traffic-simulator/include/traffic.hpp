// Traffic readings, hourly aggregation and the top-N report shared by the
// sequential, threaded and MPI versions.
//
// Input lines look like:   2025-03-01 16:05,TL07,184
// (timestamp, traffic light id, cars counted in the 5-minute window)
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace traffic {

struct Reading {
    std::string hour;   // "YYYY-MM-DD HH": sorts chronologically as text, across midnight too
    std::string light;
    long cars = 0;
};

// Returns nothing for the header, blank lines, comments and malformed lines.
inline std::optional<Reading> parse_line(const std::string& line) {
    if (line.empty() || line[0] == '#') return std::nullopt;
    const auto c1 = line.find(',');
    if (c1 == std::string::npos) return std::nullopt;
    const auto c2 = line.find(',', c1 + 1);
    if (c2 == std::string::npos) return std::nullopt;
    const std::string ts = line.substr(0, c1);
    // Expect "YYYY-MM-DD HH:MM"
    if (ts.size() != 16 || ts[4] != '-' || ts[7] != '-' || ts[10] != ' ' || ts[13] != ':') return std::nullopt;
    Reading r;
    r.hour = ts.substr(0, 13);
    r.light = line.substr(c1 + 1, c2 - c1 - 1);
    if (r.light.empty()) return std::nullopt;
    try {
        std::size_t used = 0;
        std::string cars = line.substr(c2 + 1);
        if (!cars.empty() && cars.back() == '\r') cars.pop_back();  // tolerate Windows line endings
        r.cars = std::stol(cars, &used);
        if (used != cars.size() || r.cars < 0) return std::nullopt;
    } catch (const std::exception&) {  // invalid_argument AND out_of_range
        return std::nullopt;
    }
    return r;
}

// hour -> (light -> cars). std::map keeps hours in chronological order.
using HourlyCounts = std::map<std::string, std::map<std::string, long>>;

inline void add(HourlyCounts& c, const Reading& r) { c[r.hour][r.light] += r.cars; }

inline void merge(HourlyCounts& into, const HourlyCounts& from) {
    for (const auto& [hour, lights] : from)
        for (const auto& [light, cars] : lights) into[hour][light] += cars;
}

// Text encoding for MPI: one "hour|light|cars" record per line.
inline std::string serialize(const HourlyCounts& c) {
    std::string out;
    for (const auto& [hour, lights] : c)
        for (const auto& [light, cars] : lights) out += hour + "|" + light + "|" + std::to_string(cars) + "\n";
    return out;
}

inline HourlyCounts deserialize(const std::string& s) {
    HourlyCounts c;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) {
        const auto p1 = line.find('|'), p2 = line.rfind('|');
        if (p1 == std::string::npos || p2 == p1) continue;
        c[line.substr(0, p1)][line.substr(p1 + 1, p2 - p1 - 1)] += std::stol(line.substr(p2 + 1));
    }
    return c;
}

// Top N lights per hour, ties broken by light id so the output is stable.
inline std::string format_top_n(const HourlyCounts& c, int n) {
    std::ostringstream os;
    for (const auto& [hour, lights] : c) {
        std::vector<std::pair<std::string, long>> v(lights.begin(), lights.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });
        os << hour << ":00 ";
        for (int i = 0; i < n && i < static_cast<int>(v.size()); ++i)
            os << (i ? ", " : "") << v[i].first << "=" << v[i].second;
        os << "\n";
    }
    return os.str();
}

inline std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(std::move(line));
    return lines;
}

struct Args {
    std::string file = "data/sample.csv";
    int top = 3;
    int producers = 2;
    int consumers = 2;
    int queue = 256;
    int batch = 1;            // lines per queue item (threads version)
    int sensor_delay_us = 0;  // simulate sensors that produce readings in real time
    bool quiet = false;       // print only the summary lines, not the hourly report
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << s << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (s == "--file") a.file = next();
        else if (s == "--top") a.top = std::stoi(next());
        else if (s == "--producers") a.producers = std::stoi(next());
        else if (s == "--consumers") a.consumers = std::stoi(next());
        else if (s == "--queue") a.queue = std::stoi(next());
        else if (s == "--batch") a.batch = std::stoi(next());
        else if (s == "--sensor-delay-us") a.sensor_delay_us = std::stoi(next());
        else if (s == "--quiet") a.quiet = true;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--file CSV] [--top N] [--producers P] [--consumers C] [--queue Q] [--batch B]\n"
                         "       [--sensor-delay-us US] [--quiet]\n";
            std::exit(s == "--help" ? 0 : 1);
        }
    }
    if (a.top < 1 || a.producers < 1 || a.consumers < 1 || a.queue < 1 || a.batch < 1 || a.sensor_delay_us < 0) {
        std::cerr << "invalid arguments\n";
        std::exit(1);
    }
    return a;
}

// Every program prints the report (unless --quiet), then the same summary lines.
inline void print_result(const std::string& version, const Args& a, const HourlyCounts& c, long readings,
                         long skipped, double seconds) {
    const std::string report = format_top_n(c, a.top);
    if (!a.quiet) std::cout << report;
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : report) { h ^= ch; h *= 1099511628211ULL; }
    std::cout << "version: " << version << "\nhours: " << c.size() << "\nreadings: " << readings
              << "\nskipped_lines: " << skipped << "\nreport_checksum: " << h << "\nseconds: " << seconds << "\n";
}

}  // namespace traffic
