#!/usr/bin/env bash
# Every version must output exactly what std::sort outputs for the same input
# (--verify), including sizes that do not divide evenly between ranks, tiny
# inputs, all-equal values and mostly distinct values.
# Set SKIP_OPENCL=1 if you have no OpenCL runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
MPIRUN="${MPIRUN:-mpirun}"
MPI_EXTRA="${MPI_EXTRA:-}"
fail=0

run() {
    local name="$1"; shift
    local out
    if ! out=$("$@" 2>&1); then echo "FAIL  $name (exit code)"; echo "$out"; fail=1; return; fi
    if grep -q "^verify: ok" <<<"$out"; then echo "PASS  $name"; else echo "FAIL  $name"; echo "$out"; fail=1; fi
}

for case in "0 10" "1 10" "17 1" "1001 1000" "100003 1000" "100003 2000000000"; do
    set -- $case; n=$1; range=$2
    tag="n=$n range=$range"
    for algo in lomuto 3way std; do
        run "sequential $algo $tag" ./bin/sequential --n $n --range $range --algo $algo --verify
    done
    run "threads t=4 $tag" ./bin/threads --n $n --range $range --threads 4 --verify
    for r in 1 3; do
        run "mpi ranks=$r $tag" $MPIRUN $MPI_EXTRA -n $r ./bin/mpi --n $n --range $range --verify
    done
    if [[ -z "${SKIP_OPENCL:-}" ]]; then
        run "mpi_opencl bitonic ranks=3 $tag" $MPIRUN $MPI_EXTRA -n 3 ./bin/mpi_opencl --n $n --range $range --kernel bitonic --verify
        if (( n <= 1001 )); then  # odd-even is O(n^2): keep it small
            run "mpi_opencl oddeven ranks=3 $tag" $MPIRUN $MPI_EXTRA -n 3 ./bin/mpi_opencl --n $n --range $range --kernel oddeven --verify
        fi
    fi
done

[[ $fail -eq 0 ]] && echo "All versions match std::sort." || { echo "Mismatches found."; exit 1; }
