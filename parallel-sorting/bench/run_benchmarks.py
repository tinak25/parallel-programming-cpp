#!/usr/bin/env python3
"""Times the sorting programs; writes results/benchmarks.csv.

Experiments:
  duplicates : Lomuto vs 3-way vs std::sort on values 0-999 (the original
               assignment data) as N grows, plus one run on distinct values.
  parallel   : threads and MPI on 10 million mostly distinct values.
  opencl     : odd-even transposition vs bitonic sort vs the CPU.
Every run's output checksum must match std::sort's.
"""
import csv
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
REPEATS = int(os.environ.get("REPEATS", "3"))
MPIRUN = os.environ.get("MPIRUN", "mpirun").split()
MPI_EXTRA = os.environ.get("MPI_EXTRA", "").split()
SKIP_OPENCL = bool(os.environ.get("SKIP_OPENCL"))


def run(cmd):
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True).stdout
    vals = dict(l.split(": ", 1) for l in out.splitlines() if ": " in l)
    return float(vals["seconds"]), vals["checksum"]


def main():
    rows, reference = [], {}

    def record(experiment, label, n, rng, cmd, repeats=REPEATS):
        key = (n, rng)
        if key not in reference:
            reference[key] = run([BIN / "sequential", "--n", n, "--range", rng, "--algo", "std"])[1]
        times = []
        for _ in range(repeats):
            t, s = run(cmd)
            assert s == reference[key], f"{label} n={n} range={rng}: wrong output"
            times.append(t)
        secs = statistics.median(times)
        rows.append(dict(experiment=experiment, label=label, n=n, range=rng, seconds=round(secs, 6)))
        print(f"{experiment:10s} {label:30s} n={n:>9,} range={rng:>10,}  {secs:8.3f}s", flush=True)

    # 1. Repeated values
    for n in [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]:
        for algo in ["lomuto", "3way", "std"]:
            record("duplicates", f"sequential {algo}", n, 1000,
                   [BIN / "sequential", "--n", n, "--range", 1000, "--algo", algo])
    for algo in ["lomuto", "3way", "std"]:
        record("distinct", f"sequential {algo}", 4_000_000, 2_000_000_000,
               [BIN / "sequential", "--n", 4_000_000, "--range", 2_000_000_000, "--algo", algo])

    # 2. Threads and MPI
    n, rng = 10_000_000, 2_000_000_000
    record("parallel", "sequential 3way", n, rng, [BIN / "sequential", "--n", n, "--range", rng, "--algo", "3way"])
    for t in [2, 4]:
        record("parallel", f"threads t={t}", n, rng, [BIN / "threads", "--n", n, "--range", rng, "--threads", t])
    for r in [1, 2]:
        record("parallel", f"mpi ranks={r}", n, rng, [*MPIRUN, *MPI_EXTRA, "-n", r, BIN / "mpi", "--n", n, "--range", rng])

    # 3. OpenCL kernels (1 rank, so it is purely the kernel vs the CPU)
    if not SKIP_OPENCL:
        for n in [10_000, 50_000]:
            record("opencl", "opencl odd-even", n, 1000,
                   [*MPIRUN, *MPI_EXTRA, "-n", 1, BIN / "mpi_opencl", "--n", n, "--kernel", "oddeven"], repeats=1)
        for n in [10_000, 50_000, 1_000_000]:
            record("opencl", "opencl bitonic", n, 1000,
                   [*MPIRUN, *MPI_EXTRA, "-n", 1, BIN / "mpi_opencl", "--n", n, "--kernel", "bitonic"])
            record("opencl", "cpu 3way", n, 1000, [BIN / "sequential", "--n", n, "--algo", "3way"])

    out = ROOT / "results" / "benchmarks.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write(f"# {platform.system()} {platform.machine()}, {os.cpu_count()} logical cores, "
                f"median of {REPEATS} runs (odd-even: 1 run)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
