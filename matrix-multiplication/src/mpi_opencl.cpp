// Hybrid: MPI splits rows between ranks, each rank multiplies its block on an
// OpenCL device. On one machine every rank shares the same device, so this
// only makes sense on a cluster where each node has its own GPU; it is
// included to show the structure.
#include "cl_matmul.hpp"
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
    const ClTiming t = cl_multiply(local_a.data(), B.data(), local_c.data(), a.n, s.my_rows);
    collect(a.n, rank, s, local_c, C);
    MPI_Barrier(MPI_COMM_WORLD);
    const double total = MPI_Wtime() - t0;

    // Report the slowest rank's setup time separately, and the rest as compute.
    double setup = t.setup, max_setup = 0;
    MPI_Reduce(&setup, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        std::cout << "device: " << t.device << "\nsetup_seconds: " << max_setup << "\n";
        rc = report("mpi_opencl-r" + std::to_string(ranks), a, C, total - max_setup, &A, &B);
    }
    MPI_Finalize();
    return rc;
}
