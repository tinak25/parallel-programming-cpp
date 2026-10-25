// MPI master-worker: rank 0 reads the file and scatters blocks of lines;
// every rank (rank 0 included) parses and aggregates its block; rank 0
// gathers the partial hourly counts and merges them.
//
// Messages are sized at run time: rank 0 scatters/gathers the byte counts
// first, then the data (MPI_Scatterv / MPI_Gatherv). The original version
// received into a fixed char buffer[4096] and aborted with "Message
// truncated" on any realistic file.
#include <mpi.h>

#include "traffic.hpp"

int main(int argc, char** argv) {
    using namespace traffic;
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const Args a = parse_args(argc, argv);

    std::vector<std::string> lines;
    if (rank == 0) lines = read_lines(a.file);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    // 1. Rank 0 packs each rank's block of whole lines into one byte array.
    std::string packed;
    std::vector<int> bytes(ranks, 0), offsets(ranks, 0);
    if (rank == 0) {
        const std::size_t total = lines.size();
        for (int r = 0; r < ranks; ++r) {
            offsets[r] = static_cast<int>(packed.size());
            for (std::size_t i = total * r / ranks; i < total * (r + 1) / ranks; ++i) packed += lines[i] + "\n";
            bytes[r] = static_cast<int>(packed.size()) - offsets[r];
        }
    }
    int my_bytes = 0;
    MPI_Scatter(bytes.data(), 1, MPI_INT, &my_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::string block(static_cast<std::size_t>(my_bytes), '\0');
    MPI_Scatterv(packed.data(), bytes.data(), offsets.data(), MPI_CHAR, block.data(), my_bytes, MPI_CHAR, 0,
                 MPI_COMM_WORLD);

    // 2. Every rank aggregates its block.
    HourlyCounts local;
    long counters[2] = {0, 0};  // readings, skipped
    std::size_t start = 0;
    while (start < block.size()) {
        std::size_t end = block.find('\n', start);
        if (end == std::string::npos) end = block.size();
        if (auto r = parse_line(block.substr(start, end - start))) {
            add(local, *r);
            ++counters[0];
        } else {
            ++counters[1];
        }
        start = end + 1;
    }

    // 3. Gather the partial results (sizes first, then data) and merge.
    const std::string mine = serialize(local);
    int mine_bytes = static_cast<int>(mine.size());
    std::vector<int> rbytes(ranks), roffsets(ranks);
    MPI_Gather(&mine_bytes, 1, MPI_INT, rbytes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::string all;
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < ranks; ++r) { roffsets[r] = off; off += rbytes[r]; }
        all.resize(static_cast<std::size_t>(off));
    }
    MPI_Gatherv(mine.data(), mine_bytes, MPI_CHAR, all.data(), rbytes.data(), roffsets.data(), MPI_CHAR, 0,
                MPI_COMM_WORLD);
    long totals[2] = {0, 0};
    MPI_Reduce(counters, totals, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        HourlyCounts counts;
        for (int r = 0; r < ranks; ++r)
            merge(counts, deserialize(all.substr(static_cast<std::size_t>(roffsets[r]), static_cast<std::size_t>(rbytes[r]))));
        const double secs = MPI_Wtime() - t0;
        print_result("mpi-r" + std::to_string(ranks), a, counts, totals[0], totals[1], secs);
        int largest = 0;
        for (int b : rbytes) largest = std::max(largest, b);
        std::cout << "largest_result_message_bytes: " << largest << "\n";
    }
    MPI_Finalize();
    return 0;
}
