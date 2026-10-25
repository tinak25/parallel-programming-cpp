// Shared by the MPI sorting programs: split N values into one chunk per rank
// (Scatterv, so N does not have to divide evenly), then gather the sorted
// chunks and merge them on rank 0.
//
// The original coursework used MPI_Scatter with chunk = N / size, which
// silently dropped the last N % size values.
#pragma once

#include <mpi.h>

#include "sorting.hpp"

namespace srt {

struct Chunks {
    std::vector<int> counts, offsets;
    int mine = 0;
};

inline Chunks split(long long n, int ranks, int rank) {
    Chunks c;
    c.counts.resize(ranks);
    c.offsets.resize(ranks);
    long long off = 0;
    for (int r = 0; r < ranks; ++r) {
        const long long cnt = n / ranks + (r < n % ranks ? 1 : 0);
        c.counts[r] = static_cast<int>(cnt);
        c.offsets[r] = static_cast<int>(off);
        off += cnt;
    }
    c.mine = c.counts[rank];
    return c;
}

inline std::vector<int> scatter(const std::vector<int>& all, const Chunks& c, int rank) {
    std::vector<int> local(c.mine);
    MPI_Scatterv(rank == 0 ? all.data() : nullptr, c.counts.data(), c.offsets.data(), MPI_INT, local.data(),
                 c.mine, MPI_INT, 0, MPI_COMM_WORLD);
    return local;
}

// Returns the fully merged result on rank 0 (empty elsewhere).
inline std::vector<int> gather_and_merge(const std::vector<int>& local, const Chunks& c, int rank, long long n) {
    std::vector<int> gathered(rank == 0 ? static_cast<std::size_t>(n) : 0);
    MPI_Gatherv(local.data(), c.mine, MPI_INT, rank == 0 ? gathered.data() : nullptr, c.counts.data(),
                c.offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) return {};
    return kway_merge(gathered, c.offsets, c.counts);
}

}  // namespace srt
