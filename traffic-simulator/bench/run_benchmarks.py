#!/usr/bin/env python3
"""Times the traffic programs; writes results/benchmarks.csv.

Two experiments:
  cpu     : a month of data (make data), as fast as possible. Measures how the
            parsing and aggregation work splits across threads or ranks.
  sensors : the one-day sample with every reading arriving 1 ms after the last,
            like real sensors. Measures what producer-consumer is really for:
            overlapping waiting.
Every result is checked against the sequential report.
"""
import csv
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
REPEATS = int(os.environ.get("REPEATS", "5"))
MPIRUN = os.environ.get("MPIRUN", "mpirun").split()
MPI_EXTRA = os.environ.get("MPI_EXTRA", "").split()


def run(cmd):
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True).stdout
    vals = dict(l.split(": ", 1) for l in out.splitlines() if ": " in l)
    return float(vals["seconds"]), vals["report_checksum"], vals


def main():
    rows, expected = [], {}

    def record(experiment, label, cmd, producers=0, consumers=0, ranks=1):
        times, sums, extra = [], set(), {}
        for _ in range(REPEATS):
            t, s, extra = run(cmd)
            times.append(t)
            sums.add(s)
        assert len(sums) == 1
        s = sums.pop()
        expected.setdefault(experiment, s)
        assert s == expected[experiment], f"{label} printed a different report"
        secs = statistics.median(times)
        rows.append(dict(experiment=experiment, label=label, producers=producers, consumers=consumers,
                         ranks=ranks, seconds=round(secs, 6),
                         largest_message_bytes=extra.get("largest_result_message_bytes", "")))
        print(f"{experiment:8s} {label:28s} {secs:.3f}s", flush=True)

    month = ["--file", ROOT / "data" / "month.csv", "--quiet"]
    record("cpu", "sequential", [BIN / "sequential", *month])
    for p, c, b in [(1, 1, 1), (1, 2, 1), (1, 1, 1000), (1, 2, 1000)]:
        record("cpu", f"threads p={p} c={c} batch={b}",
               [BIN / "threads", *month, "--producers", p, "--consumers", c, "--batch", b], p, c)
    for r in [1, 2]:
        record("cpu", f"mpi ranks={r}", [*MPIRUN, *MPI_EXTRA, "-n", r, BIN / "mpi", *month], ranks=r)

    sample = ["--file", ROOT / "data" / "sample.csv", "--quiet", "--sensor-delay-us", "1000"]
    record("sensors", "sequential", [BIN / "sequential", *sample])
    for p, c in [(1, 1), (2, 1), (4, 1), (8, 1), (8, 2)]:
        record("sensors", f"threads p={p} c={c}", [BIN / "threads", *sample, "--producers", p, "--consumers", c],
               p, c)

    out = ROOT / "results" / "benchmarks.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write(f"# {platform.system()} {platform.machine()}, {os.cpu_count()} logical cores, "
                f"median of {REPEATS} runs\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
