// Hybrid: MPI splits rows between ranks, OpenMP splits each rank's rows
// between threads.
#ifndef _OPENMP
#error "Compile with -fopenmp, otherwise #pragma omp is silently ignored"
#endif
#include <omp.h>

#include "mpi_rows.hpp"

int main(int argc, char** argv) {
    using namespace mm;
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  // only the main thread calls MPI
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const Args a = parse_args(argc, argv);
    omp_set_num_threads(a.threads);

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
    const int n = a.n, rows = s.my_rows;
    const bool ijk = a.order == "ijk";
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        if (ijk) multiply_rows_ijk(local_a.data(), B.data(), local_c.data(), n, i, i + 1);
        else multiply_rows_ikj(local_a.data(), B.data(), local_c.data(), n, i, i + 1);
    }
    collect(a.n, rank, s, local_c, C);
    MPI_Barrier(MPI_COMM_WORLD);
    const double secs = MPI_Wtime() - t0;

    int rc = 0;
    if (rank == 0)
        rc = report("mpi_openmp-" + a.order + "-r" + std::to_string(ranks) + "-t" + std::to_string(a.threads), a,
                    C, secs, &A, &B);
    MPI_Finalize();
    return rc;
}
