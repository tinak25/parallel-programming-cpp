// MPI + OpenCL sort: scatter chunks, each rank sorts its chunk on an OpenCL
// device, gather, k-way merge on rank 0.
//
// Two kernels:
//   oddeven : odd-even transposition sort, the one used in the original
//             coursework. It needs n passes over the data, so the total work
//             is O(n^2) no matter how many processors run each pass.
//   bitonic : bitonic sort, O(n log^2 n) work in O(log^2 n) passes. It is the
//             standard GPU sorting network.
// Rank 0 does NOT re-sort the result afterwards (the original called
// std::sort on the whole array, which hid whether the kernel had worked).
#include <climits>

#include "cl_helpers.hpp"
#include "mpi_chunks.hpp"

namespace {

const char* kSortKernels = R"CLC(
__kernel void oddeven_pass(__global int* a, const int n, const int phase) {
    const int i = 2 * get_global_id(0) + (phase & 1);
    if (i + 1 < n && a[i] > a[i + 1]) {
        const int t = a[i]; a[i] = a[i + 1]; a[i + 1] = t;
    }
}

__kernel void bitonic_step(__global int* a, const uint j, const uint k) {
    const uint i = get_global_id(0);
    const uint partner = i ^ j;
    if (partner > i) {
        const int x = a[i], y = a[partner];
        const bool ascending = (i & k) == 0;
        if ((ascending && x > y) || (!ascending && x < y)) { a[i] = y; a[partner] = x; }
    }
}
)CLC";

struct SortTiming { double setup = 0, compute = 0; std::string device; };

SortTiming cl_sort(std::vector<int>& v, const std::string& kernel) {
    SortTiming t;
    auto t0 = std::chrono::steady_clock::now();
    clh::Device d = clh::open_device();
    cl_program prog = nullptr;
    cl_kernel k = clh::build_kernel(d, kSortKernels, kernel == "oddeven" ? "oddeven_pass" : "bitonic_step", &prog);
    t.setup = srt::seconds_since(t0);
    t.device = d.name;

    t0 = std::chrono::steady_clock::now();
    const std::size_t n = v.size();
    if (n > 1) {
        // Bitonic sort needs a power-of-two length: pad with INT_MAX, which sorts to the end.
        std::size_t m = n;
        if (kernel == "bitonic") { m = 1; while (m < n) m <<= 1; }
        std::vector<int> work(v);
        work.resize(m, INT_MAX);

        cl_int err = CL_SUCCESS;
        cl_mem buf = clCreateBuffer(d.context, CL_MEM_READ_WRITE, sizeof(int) * m, nullptr, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(d.queue, buf, CL_TRUE, 0, sizeof(int) * m, work.data(), 0, nullptr, nullptr));
        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &buf));

        if (kernel == "oddeven") {
            const int len = static_cast<int>(n);
            const std::size_t global = std::max<std::size_t>(1, n / 2);
            CL_CHECK(clSetKernelArg(k, 1, sizeof(int), &len));
            for (int phase = 0; phase < len; ++phase) {  // n passes: this is the O(n^2) part
                CL_CHECK(clSetKernelArg(k, 2, sizeof(int), &phase));
                CL_CHECK(clEnqueueNDRangeKernel(d.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
            }
        } else {
            for (cl_uint kk = 2; kk <= m; kk <<= 1) {
                for (cl_uint j = kk >> 1; j > 0; j >>= 1) {
                    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_uint), &j));
                    CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_uint), &kk));
                    CL_CHECK(clEnqueueNDRangeKernel(d.queue, k, 1, nullptr, &m, nullptr, 0, nullptr, nullptr));
                }
            }
        }
        CL_CHECK(clEnqueueReadBuffer(d.queue, buf, CL_TRUE, 0, sizeof(int) * n, v.data(), 0, nullptr, nullptr));
        CL_CHECK(clFinish(d.queue));
        clReleaseMemObject(buf);
    }
    t.compute = srt::seconds_since(t0);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clh::release(d);
    return t;
}

}  // namespace

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
    const SortTiming st = cl_sort(local, a.kernel);
    const std::vector<int> sorted = gather_and_merge(local, c, rank, a.n);
    const double total = MPI_Wtime() - t0;

    double setup = st.setup, max_setup = 0;
    MPI_Reduce(&setup, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        std::cout << "device: " << st.device << "\nsetup_seconds: " << max_setup << "\n";
        rc = report("mpi_opencl-" + a.kernel + "-r" + std::to_string(ranks), a, input, sorted, total - max_setup);
    }
    MPI_Finalize();
    return rc;
}
