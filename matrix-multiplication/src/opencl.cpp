// OpenCL on a single device (GPU if there is one, otherwise CPU).
// `seconds` includes copying the matrices to and from the device, not only
// the kernel, so it is comparable with the CPU versions.
#include "cl_matmul.hpp"

int main(int argc, char** argv) {
    using namespace mm;
    const Args a = parse_args(argc, argv);
    const Matrix A = random_matrix(a.n, a.seed), B = random_matrix(a.n, a.seed + 1);
    Matrix C(A.size());

    const ClTiming t = cl_multiply(A.data(), B.data(), C.data(), a.n, a.n);
    std::cout << "device: " << t.device << "\nsetup_seconds: " << t.setup << "\n";
    return report("opencl", a, C, t.compute, &A, &B);
}
