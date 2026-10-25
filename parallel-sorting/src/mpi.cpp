// MPI sort: scatter chunks, each rank sorts its chunk, gather, k-way merge
// on rank 0. The timer covers all of it, including the merge (the original
// stopped the clock before the merge).
#include "mpi_chunks.hpp"

int main(int argc, char** argv) {
    using namespace srt;
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const Args a = parse_args(argc, argv);

    std::vector<int> input;
    if (rank == 0) input = make_data(a.n, a.range, a.seed);
    const Chunks c = split(a.n, ranks, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    std::vector<int> local = scatter(input, c, rank);
    sort_with(a.algo, local);
    const std::vector<int> sorted = gather_and_merge(local, c, rank, a.n);
    const double secs = MPI_Wtime() - t0;

    int rc = 0;
    if (rank == 0) rc = report("mpi-" + a.algo + "-r" + std::to_string(ranks), a, input, sorted, secs);
    MPI_Finalize();
    return rc;
}
