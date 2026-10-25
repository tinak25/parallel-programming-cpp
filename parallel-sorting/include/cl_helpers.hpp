// Minimal OpenCL setup with every call checked.
//
// The original coursework version ignored every OpenCL return code, which hid
// a real bug in the matrix version (a 5th kernel argument set on a 4-argument kernel). CL_CHECK
// stops the program with the failing call's name instead.
#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CL_CHECK(call)                                                                       \
    do {                                                                                     \
        cl_int _err = (call);                                                                \
        if (_err != CL_SUCCESS) {                                                            \
            std::cerr << "OpenCL error " << _err << " in " #call " (" << __FILE__ << ":" << __LINE__ \
                      << ")\n";                                                              \
            std::exit(3);                                                                    \
        }                                                                                    \
    } while (0)

namespace clh {

struct Device {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string name;
};

// Prefers a GPU, falls back to any device (e.g. the POCL CPU runtime in Docker).
inline Device open_device() {
    Device d;
    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
    if (np == 0) {
        std::cerr << "no OpenCL platform found\n";
        std::exit(3);
    }
    std::vector<cl_platform_id> platforms(np);
    CL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr));
    for (cl_device_type type : {static_cast<cl_device_type>(CL_DEVICE_TYPE_GPU),
                                static_cast<cl_device_type>(CL_DEVICE_TYPE_ALL)}) {
        for (auto p : platforms) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(p, type, 1, &d.device, &nd) == CL_SUCCESS && nd > 0) {
                d.platform = p;
                break;
            }
        }
        if (d.platform) break;
    }
    if (!d.platform) {
        std::cerr << "no OpenCL device found\n";
        std::exit(3);
    }
    char buf[256] = {0};
    clGetDeviceInfo(d.device, CL_DEVICE_NAME, sizeof(buf) - 1, buf, nullptr);
    d.name = buf;
    cl_int err = CL_SUCCESS;
    d.context = clCreateContext(nullptr, 1, &d.device, nullptr, nullptr, &err);
    CL_CHECK(err);
    d.queue = clCreateCommandQueue(d.context, d.device, 0, &err);
    CL_CHECK(err);
    return d;
}

inline cl_kernel build_kernel(const Device& d, const char* source, const char* name, cl_program* out_program) {
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(d.context, 1, &source, nullptr, &err);
    CL_CHECK(err);
    if (clBuildProgram(prog, 1, &d.device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        std::size_t len = 0;
        clGetProgramBuildInfo(prog, d.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, d.device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::cerr << "kernel build failed:\n" << log << "\n";
        std::exit(3);
    }
    cl_kernel k = clCreateKernel(prog, name, &err);
    CL_CHECK(err);
    *out_program = prog;
    return k;
}

inline void release(Device& d) {
    if (d.queue) clReleaseCommandQueue(d.queue);
    if (d.context) clReleaseContext(d.context);
}

}  // namespace clh
