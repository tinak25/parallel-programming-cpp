#!/usr/bin/env bash
# Runs every version on sizes that do and do not divide evenly between
# threads/ranks, and checks that each one produces exactly the same matrix
# as the single-threaded reference (element by element, via --verify).
# Set SKIP_OPENCL=1 if you have no OpenCL runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
MPIRUN="${MPIRUN:-mpirun}"
MPI_EXTRA="${MPI_EXTRA:-}"
fail=0

run() {
    local name="$1"; shift
    local out
    if ! out=$("$@" 2>&1); then
        echo "FAIL  $name (exit code)"; echo "$out"; fail=1; return
    fi
    if grep -q "^verify: ok" <<<"$out"; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"; echo "$out"; fail=1
    fi
}

for n in 1 97 256; do
    for order in ijk ikj; do
        run "sequential  n=$n $order" ./bin/sequential --n $n --order $order --verify
        run "pthreads    n=$n $order t=3" ./bin/pthreads --n $n --order $order --threads 3 --verify
        run "openmp      n=$n $order t=3" ./bin/openmp --n $n --order $order --threads 3 --verify
    done
    for r in 1 3; do
        run "mpi         n=$n ranks=$r" $MPIRUN $MPI_EXTRA -n $r ./bin/mpi --n $n --verify
        run "mpi_openmp  n=$n ranks=$r t=2" $MPIRUN $MPI_EXTRA -n $r ./bin/mpi_openmp --n $n --threads 2 --verify
    done
    if [[ -z "${SKIP_OPENCL:-}" ]]; then
        run "opencl      n=$n" ./bin/opencl --n $n --verify
        run "mpi_opencl  n=$n ranks=3" $MPIRUN $MPI_EXTRA -n 3 ./bin/mpi_opencl --n $n --verify
    fi
done

[[ $fail -eq 0 ]] && echo "All versions match the reference." || { echo "Mismatches found."; exit 1; }
