// Multiplies `rows` rows of A by B on an OpenCL device.
// Returns the time spent copying data and running the kernel; device setup
// and kernel compilation are reported separately because they are one-off
// costs that a long-running service would pay only once.
#pragma once

#include <chrono>

#include "cl_helpers.hpp"
#include "matmul_kernel.hpp"
#include "matrix.hpp"

namespace mm {

struct ClTiming {
    double setup = 0;    // device, context, kernel compilation
    double compute = 0;  // copy in + kernel + copy out
    std::string device;
};

inline ClTiming cl_multiply(const int* a, const int* b, int* c, int n, int rows) {
    ClTiming t;
    auto t0 = std::chrono::steady_clock::now();
    clh::Device d = clh::open_device();
    cl_program prog = nullptr;
    cl_kernel k = clh::build_kernel(d, kMatmulKernel, "matmul", &prog);
    t.setup = seconds_since(t0);
    t.device = d.name;

    t0 = std::chrono::steady_clock::now();
    cl_int err = CL_SUCCESS;
    const std::size_t a_bytes = sizeof(int) * static_cast<std::size_t>(rows) * n;
    const std::size_t b_bytes = sizeof(int) * static_cast<std::size_t>(n) * n;
    cl_mem da = clCreateBuffer(d.context, CL_MEM_READ_ONLY, a_bytes ? a_bytes : 4, nullptr, &err);
    CL_CHECK(err);
    cl_mem db = clCreateBuffer(d.context, CL_MEM_READ_ONLY, b_bytes, nullptr, &err);
    CL_CHECK(err);
    cl_mem dc = clCreateBuffer(d.context, CL_MEM_WRITE_ONLY, a_bytes ? a_bytes : 4, nullptr, &err);
    CL_CHECK(err);
    if (rows > 0) CL_CHECK(clEnqueueWriteBuffer(d.queue, da, CL_TRUE, 0, a_bytes, a, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(d.queue, db, CL_TRUE, 0, b_bytes, b, 0, nullptr, nullptr));

    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &da));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &db));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &dc));
    CL_CHECK(clSetKernelArg(k, 3, sizeof(int), &n));
    CL_CHECK(clSetKernelArg(k, 4, sizeof(int), &rows));
    if (rows > 0) {
        const std::size_t global[2] = {static_cast<std::size_t>(rows), static_cast<std::size_t>(n)};
        CL_CHECK(clEnqueueNDRangeKernel(d.queue, k, 2, nullptr, global, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(d.queue, dc, CL_TRUE, 0, a_bytes, c, 0, nullptr, nullptr));
    }
    CL_CHECK(clFinish(d.queue));
    t.compute = seconds_since(t0);

    clReleaseMemObject(da);
    clReleaseMemObject(db);
    clReleaseMemObject(dc);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clh::release(d);
    return t;
}

}  // namespace mm
