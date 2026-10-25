// MPI: each rank multiplies a block of rows on one thread.
// The timer covers the communication too (scatter, broadcast, gather),
// because that is part of the real cost of distributing the work.
#include "mpi_rows.hpp"

int main(int argc, char** argv) {
    using namespace mm;
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const Args a = parse_args(argc, argv);

    Matrix A, B, local_a, C;
    if (rank == 0) {
        A = random_matrix(a.n, a.seed);
        B = random_matrix(a.n, a.seed + 1);
    }
    const RowSplit s = split_rows(a.n, ranks, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    distribute(a.n, rank, s, A, B, local_a);
    Matrix local_c(local_a.size());
    // local rows are numbered 0..my_rows-1 within local_a / local_c
    multiply_rows(a.order, local_a.data(), B.data(), local_c.data(), a.n, 0, s.my_rows);
    collect(a.n, rank, s, local_c, C);
    MPI_Barrier(MPI_COMM_WORLD);
    const double secs = MPI_Wtime() - t0;

    int rc = 0;
    if (rank == 0) rc = report("mpi-" + a.order + "-r" + std::to_string(ranks), a, C, secs, &A, &B);
    MPI_Finalize();
    return rc;
}
