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
        r["n"] = int(r["n"])
    return machine, rows


def duplicates_chart(machine, rows):
    data = [r for r in rows if r["experiment"] == "duplicates"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, name, color in [("sequential lomuto", "Lomuto partition (original)", ORANGE),
                               ("sequential 3way", "3-way partition (fix)", GREEN),
                               ("sequential std", "std::sort (dashed, almost identical to 3-way)", BLUE)]:
        pts = sorted((r["n"], r["seconds"]) for r in data if r["label"] == label)
        xs, ys = [p[0] / 1e6 for p in pts], [p[1] for p in pts]
        dashed = label == "sequential std"
        ax.plot(xs, ys, marker="o" if not dashed else None, color=color, linewidth=1.5 if dashed else 2.5,
                linestyle="--" if dashed else "-", label=name, zorder=3 if dashed else 2)
        if not dashed:
            ax.text(xs[-1] + 0.08, ys[-1], f"{ys[-1]:.2f}s", va="center", fontsize=9, color=color)
    ax.set_xlabel("Number of values (millions), each between 0 and 999")
    ax.set_ylabel("Seconds (lower is better)")
    ax.set_xlim(0, max(r["n"] for r in data) / 1e6 * 1.15)
    ax.set_title("Repeated values turn the original quicksort quadratic")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.01, f"Single thread. Measured on {machine}.", color=MUTED, fontsize=8)
    fig.savefig(ROOT / "charts" / "duplicates.png", dpi=160)
    plt.close(fig)


def bars(machine, rows, experiment, filename, title, xlabel, note, color_of):
    data = [r for r in rows if r["experiment"] == experiment]
    base = data[0]["seconds"]
    labels = [r["label"] for r in data][::-1]
    secs = [r["seconds"] for r in data][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(data) + 1.7))
    ax.grid(axis="y", visible=False)
    b = ax.barh(labels, secs, color=[color_of(l) for l in labels], height=0.6)
    for bar, s in zip(b, secs):
        ax.text(s + max(secs) * 0.01, bar.get_y() + bar.get_height() / 2, f"{s:.3f}s  ({base / s:.2f}x)",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(secs) * 1.32)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, f"{note} Measured on {machine}.", color=MUTED, fontsize=8)
    fig.savefig(ROOT / "charts" / filename, dpi=160)
    plt.close(fig)


def opencl_chart(machine, rows):
    data = [r for r in rows if r["experiment"] == "opencl"]
    sizes = sorted({r["n"] for r in data})
    series = [("opencl odd-even", "OpenCL odd-even transposition (original)", ORANGE),
              ("opencl bitonic", "OpenCL bitonic sort", BLUE), ("cpu 3way", "CPU 3-way quicksort", GREEN)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.26
    for i, (label, name, color) in enumerate(series):
        for j, n in enumerate(sizes):
            hit = [r["seconds"] for r in data if r["label"] == label and r["n"] == n]
            if not hit:
                ax.text(j + (i - 1) * width, 0.002, "too slow\nto run", ha="center", va="bottom", fontsize=7,
                        color=MUTED)
                continue
            ax.bar(j + (i - 1) * width, hit[0], width * 0.92, color=color, label=name if j == 0 else None)
            ax.text(j + (i - 1) * width, hit[0] * 1.15, f"{hit[0]:.3f}s", ha="center", va="bottom", fontsize=8,
                    color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)), [f"{n:,} values" for n in sizes])
    ax.set_ylim(top=max(r["seconds"] for r in data) * 4)
    ax.set_ylabel("Seconds, log scale (lower is better)")
    ax.set_title("The algorithm matters more than the hardware")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, "OpenCL ran on the CPU (no GPU here); kernel compilation excluded. "
                         f"{machine.split(', median')[0]}.", color=MUTED, fontsize=8)
    fig.savefig(ROOT / "charts" / "opencl_algorithms.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "charts").mkdir(exist_ok=True)
    machine, rows = load()
    duplicates_chart(machine, rows)
    bars(machine, rows, "parallel", "parallel.png", "10 million distinct values: threads vs MPI",
         "Seconds (lower is better); speed-up vs sequential in brackets",
         "MPI time includes scatter, gather and the final merge.",
         lambda l: MUTED if l.startswith("sequential") else (GREEN if l.startswith("mpi") else BLUE))
    opencl_chart(machine, rows)
    print("charts written")
