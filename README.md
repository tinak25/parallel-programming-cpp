# Parallel programming in C++: threads, OpenMP, MPI and OpenCL

Three problems, each solved from a single-threaded baseline up to distributed and hybrid versions. Every version is tested against the baseline, and every speed-up is measured, not assumed.

| Project | Threads (one machine) | Distributed / accelerated | Headline finding |
|---|---|---|---|
| [**matrix-multiplication**](matrix-multiplication) | pthreads, OpenMP | MPI, MPI + OpenMP, OpenCL, MPI + OpenCL | Reordering two loops made one thread 1.3x to 8.4x faster, and the 8.4x turned out to be a cache effect specific to N = 1024 |
| [**parallel-sorting**](parallel-sorting) | quicksort with `std::thread` | MPI, MPI + OpenCL | Repeated values made my original quicksort quadratic (6.5 s vs 0.19 s); odd-even sort on OpenCL was 180x slower than bitonic sort |
| [**traffic-simulator**](traffic-simulator) | producer-consumer with a bounded queue | MPI master-worker | 8x from overlapping sensor waits, but 7.7x *slower* when every line went through the queue, until batching fixed it |

These started as assignments for a university course on concurrent and distributed programming. I rebuilt them after reviewing my original code, which had several real bugs:
- a quicksort that went quadratic on repeated values
- MPI scatters that silently dropped values
- a fixed 4 KB message buffer that crashed on realistic data
- a hybrid MPI + OpenMP version whose timings match OpenMP being silently off (a missing compiler flag does exactly that)
- OpenCL errors that were never checked
- speed-ups measured from single runs, some probably on unoptimised builds

Each project's README lists what was wrong and how I fixed it. My final project for the same course, a hybrid MPI and threads airport simulator, has [its own repository](https://github.com/tinak25/airport-flight-simulator).

## Skills shown

- **Shared memory:** POSIX threads, `std::thread`, OpenMP, mutexes, condition variables, a bounded blocking queue, lock scope, load balancing
- **Distributed memory:** MPI collectives (`Scatter`, `Scatterv`, `Gatherv`, `Bcast`, `Reduce`), variable-size messages, master-worker and data-parallel decomposition
- **Accelerators:** OpenCL host code with error checking, matrix multiplication, odd-even and bitonic sorting kernels
- **Performance engineering:** cache-aware loop ordering, task granularity and batching, reproducible benchmarks (fixed seeds, medians, correctness checks before timing)

## Build and test everything

**Docker** (nothing else to install; uses a CPU OpenCL runtime):

```bash
docker build -t parallel-cpp .
docker run --rm parallel-cpp          # builds and runs all three test suites
```

**Locally**, from inside a project folder:

```bash
make check
```

You need a C++17 compiler with OpenMP, an MPI implementation and, for the OpenCL programs, an OpenCL runtime.
- **Ubuntu:** `sudo apt install g++ make openmpi-bin libopenmpi-dev opencl-headers ocl-icd-opencl-dev pocl-opencl-icd python3-matplotlib`
- **macOS:** `brew install open-mpi libomp`. OpenCL is built in, and the matrix README explains the OpenMP flags.
- **No OpenCL?** Build with `make cpu mpi`, and run the tests with `SKIP_OPENCL=1 ./tests/check.sh`.

All benchmark numbers in the READMEs come from a 2-core Linux cloud machine with no GPU, so 2x is the ceiling for CPU speed-ups. Each project's `bench/` folder reruns the measurements on your own hardware.
