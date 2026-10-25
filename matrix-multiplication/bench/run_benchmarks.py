#!/usr/bin/env python3
"""Times every matrix multiplication version; writes results/benchmarks.csv.

Median of REPEATS runs (default 3). Every run is also checked against the
sequential checksum, so a fast-but-wrong version cannot sneak into the chart.
Run from matrix-multiplication/ after `make all`.
"""
import csv
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
OUT = ROOT / "results" / "benchmarks.csv"
REPEATS = int(os.environ.get("REPEATS", "3"))
MPIRUN = os.environ.get("MPIRUN", "mpirun").split()
MPI_EXTRA = os.environ.get("MPI_EXTRA", "").split()
SKIP_OPENCL = bool(os.environ.get("SKIP_OPENCL"))


def run(cmd):
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True).stdout
    vals = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    return float(vals["seconds"]), vals["checksum"]


def main():
    rows, reference = [], {}

    def record(experiment, label, n, cmd, ranks=1, threads=1):
        times, sums = [], set()
        for _ in range(REPEATS):
            t, s = run(cmd)
            times.append(t)
            sums.add(s)
        assert len(sums) == 1
        s = sums.pop()
        reference.setdefault(n, s)
        assert s == reference[n], f"{label} n={n} gave a different matrix"
        secs = statistics.median(times)
        rows.append(dict(experiment=experiment, label=label, n=n, ranks=ranks, threads=threads,
                         seconds=round(secs, 6)))
        print(f"{experiment:9s} {label:28s} n={n:5d}  {secs:8.3f}s", flush=True)

    # 1. Loop order and compiler optimisation, single thread
    for n in [256, 512, 1000, 1024, 1500, 1536]:
        for order in ["ijk", "ikj"]:
            record("order", f"sequential {order}", n, [BIN / "sequential", "--n", n, "--order", order])
    record("order", "sequential ijk -O0", 1000, [BIN / "sequential_O0", "--n", 1000, "--order", "ijk"])

    # 2. Everything at n = 1000 (not a power of two, see README)
    n = 1000
    for order in ["ijk", "ikj"]:
        for t in [1, 2]:
            record("all", f"pthreads {order}", n, [BIN / "pthreads", "--n", n, "--order", order, "--threads", t],
                   threads=t)
            record("all", f"openmp {order}", n, [BIN / "openmp", "--n", n, "--order", order, "--threads", t],
                   threads=t)
    for r in [1, 2]:
        record("all", "mpi ikj", n, [*MPIRUN, *MPI_EXTRA, "-n", r, BIN / "mpi", "--n", n], ranks=r)
    record("all", "mpi+openmp ikj", n,
           [*MPIRUN, *MPI_EXTRA, "-n", 1, BIN / "mpi_openmp", "--n", n, "--threads", 2], ranks=1, threads=2)
    record("all", "mpi+openmp ikj", n,
           [*MPIRUN, *MPI_EXTRA, "-n", 2, BIN / "mpi_openmp", "--n", n, "--threads", 1], ranks=2, threads=1)
    if not SKIP_OPENCL:
        record("all", "opencl", n, [BIN / "opencl", "--n", n])
        record("all", "mpi+opencl", n, [*MPIRUN, *MPI_EXTRA, "-n", 2, BIN / "mpi_opencl", "--n", n], ranks=2)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write(f"# {platform.system()} {platform.machine()}, {os.cpu_count()} logical cores, "
                f"median of {REPEATS} runs\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
