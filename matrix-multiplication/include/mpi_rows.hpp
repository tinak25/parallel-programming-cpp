// Row distribution shared by the MPI programs: rank 0 scatters blocks of A's
// rows (Scatterv, so N does not have to divide evenly), broadcasts all of B,
// and gathers the blocks of C back.
#pragma once

#include <mpi.h>

#include <vector>

#include "matrix.hpp"

namespace mm {

struct RowSplit {
    std::vector<int> counts, offsets;  // in elements, for Scatterv/Gatherv
    int my_rows = 0;
};

inline RowSplit split_rows(int n, int ranks, int rank) {
    RowSplit s;
    s.counts.resize(ranks);
    s.offsets.resize(ranks);
    int row = 0;
    for (int r = 0; r < ranks; ++r) {
        const int rows = n / ranks + (r < n % ranks ? 1 : 0);
        s.counts[r] = rows * n;
        s.offsets[r] = row * n;
        row += rows;
        if (r == rank) s.my_rows = rows;
    }
    return s;
}

// Distributes the inputs. On return every rank has its rows of A in `local_a`
// and all of B in `b`.
inline void distribute(int n, int rank, const RowSplit& s, const Matrix& a_full, Matrix& b,
                       Matrix& local_a) {
    local_a.resize(static_cast<std::size_t>(s.my_rows) * n);
    b.resize(static_cast<std::size_t>(n) * n);
    MPI_Scatterv(rank == 0 ? a_full.data() : nullptr, s.counts.data(), s.offsets.data(), MPI_INT,
                 local_a.data(), s.my_rows * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
}

inline void collect(int n, int rank, const RowSplit& s, const Matrix& local_c, Matrix& c_full) {
    if (rank == 0) c_full.resize(static_cast<std::size_t>(n) * n);
    MPI_Gatherv(local_c.data(), s.my_rows * n, MPI_INT, rank == 0 ? c_full.data() : nullptr,
                s.counts.data(), s.offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
}

}  // namespace mm
