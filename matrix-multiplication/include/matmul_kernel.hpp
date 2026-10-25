// OpenCL kernel: one work-item computes one element of C.
// `rows` lets the MPI + OpenCL version multiply only its own block of rows.
#pragma once

inline const char* kMatmulKernel = R"CLC(
__kernel void matmul(__global const int* a, __global const int* b, __global int* c,
                     const int n, const int rows) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= rows || j >= n) return;
    int sum = 0;
    for (int k = 0; k < n; ++k) sum += a[i * n + k] * b[k * n + j];
    c[i * n + j] = sum;
}
)CLC";
