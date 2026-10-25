# Builds all three projects and runs their test suites:
#   docker build -t parallel-cpp .
#   docker run --rm parallel-cpp
# OpenCL runs on the CPU through POCL, so no GPU is needed.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make python3 openmpi-bin libopenmpi-dev \
        opencl-headers ocl-icd-opencl-dev pocl-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN make -C matrix-multiplication all && make -C parallel-sorting all && make -C traffic-simulator all

# Open MPI refuses to run as root or start more ranks than cores unless told to.
ENV MPIRUN="mpirun --allow-run-as-root" MPI_EXTRA="--oversubscribe"
CMD ["bash", "-c", "matrix-multiplication/tests/check.sh && parallel-sorting/tests/check.sh && traffic-simulator/tests/check.sh"]
