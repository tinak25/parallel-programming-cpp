#!/usr/bin/env python3
"""Draws charts/*.png from results/benchmarks.csv."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})


def load():
    with open(ROOT / "results" / "benchmarks.csv") as f:
        machine = f.readline().lstrip("# ").strip()
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seconds"] = float(r["seconds"])
        for k in ("n", "ranks", "threads"):
            r[k] = int(r[k])
    return machine, rows


def get(rows, experiment, label, n, ranks=1, threads=1):
    return next(r["seconds"] for r in rows if r["experiment"] == experiment and r["label"] == label
                and r["n"] == n and r["ranks"] == ranks and r["threads"] == threads)


def loop_order_chart(machine, rows):
    sizes = sorted({r["n"] for r in rows if r["experiment"] == "order" and r["label"] == "sequential ikj"})
    ratios = [get(rows, "order", "sequential ijk", n) / get(rows, "order", "sequential ikj", n) for n in sizes]
    colors = [ORANGE if n in (1024, 1536) else BLUE for n in sizes]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.grid(axis="x", visible=False)
    bars = ax.bar([f"{n:,}" for n in sizes], ratios, color=colors, width=0.6)
    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.1, f"{r:.1f}x", ha="center", va="bottom", color=INK)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Textbook order time / cache-friendly order time")
    ax.set_title("Same arithmetic, different loop order: how much slower is i-j-k?")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="N = 1024, 1536: column steps keep hitting the same cache sets"),
                       Patch(color=BLUE, label="other sizes")], frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.01, f"Single thread, -O2. Measured on {machine}.", color=MUTED, fontsize=8)
    fig.savefig(ROOT / "charts" / "loop_order.png", dpi=160)
    plt.close(fig)


def all_chart(machine, rows):
    n = 1000
    entries = [
        ("Sequential i-j-k, -O0 (unoptimised build)", get(rows, "order", "sequential ijk -O0", n), MUTED),
        ("Sequential i-j-k (textbook order)", get(rows, "order", "sequential ijk", n), MUTED),
        ("Sequential i-k-j (cache-friendly)", get(rows, "order", "sequential ikj", n), MUTED),
        ("pthreads i-j-k, 2 threads", get(rows, "all", "pthreads ijk", n, threads=2), BLUE),
        ("pthreads i-k-j, 2 threads", get(rows, "all", "pthreads ikj", n, threads=2), BLUE),
        ("OpenMP i-k-j, 2 threads", get(rows, "all", "openmp ikj", n, threads=2), BLUE),
        ("MPI, 2 ranks", get(rows, "all", "mpi ikj", n, ranks=2), GREEN),
        ("MPI + OpenMP, 1 rank x 2 threads", get(rows, "all", "mpi+openmp ikj", n, ranks=1, threads=2), GREEN),
        ("MPI + OpenMP, 2 ranks x 1 thread", get(rows, "all", "mpi+openmp ikj", n, ranks=2, threads=1), GREEN),
    ]
    try:
        entries += [("OpenCL (on the CPU, no GPU here)", get(rows, "all", "opencl", n), ORANGE),
                    ("MPI + OpenCL, 2 ranks (CPU)", get(rows, "all", "mpi+opencl", n, ranks=2), ORANGE)]
    except StopIteration:
        pass
    base = entries[1][1]
    labels = [e[0] for e in entries][::-1]
    secs = [e[1] for e in entries][::-1]
    colors = [e[2] for e in entries][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.grid(axis="y", visible=False)
    bars = ax.barh(labels, secs, color=colors, height=0.62)
    for b, s in zip(bars, secs):
        ax.text(s + max(secs) * 0.01, b.get_y() + b.get_height() / 2, f"{s:.2f}s  ({base / s:.1f}x)",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(secs) * 1.28)
    ax.set_xlabel("Seconds for N = 1000 (lower is better); speed-up vs textbook sequential in brackets")
    ax.set_title("Every version, same matrices, same answer")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.01, f"Measured on {machine}. MPI timings include sending the data.", color=MUTED,
             fontsize=8)
    fig.savefig(ROOT / "charts" / "all_versions.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "charts").mkdir(exist_ok=True)
    machine, rows = load()
    loop_order_chart(machine, rows)
    all_chart(machine, rows)
    print("charts written")
