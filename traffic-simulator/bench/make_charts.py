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
    return machine, rows


def hbar(machine, rows, experiment, title, xlabel, filename, note):
    data = [r for r in rows if r["experiment"] == experiment]
    base = data[0]["seconds"]
    labels = [r["label"] for r in data][::-1]
    secs = [r["seconds"] for r in data][::-1]
    colors = [MUTED if l == "sequential" else (GREEN if l.startswith("mpi") else BLUE) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(data) + 1.6))
    ax.grid(axis="y", visible=False)
    bars = ax.barh(labels, secs, color=colors, height=0.6)
    for b, s in zip(bars, secs):
        ax.text(s + max(secs) * 0.01, b.get_y() + b.get_height() / 2, f"{s:.3f}s  ({base / s:.2f}x)",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(secs) * 1.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, f"{note} Measured on {machine}.", color=MUTED, fontsize=8)
    fig.savefig(ROOT / "charts" / filename, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "charts").mkdir(exist_ok=True)
    machine, rows = load()
    hbar(machine, rows, "cpu", "A month of readings: one line per queue operation is too fine-grained",
         "Seconds (lower is better); speed-up vs sequential in brackets", "cpu_bound.png",
         "~520k readings, 60 lights, 30 days. p = producers, c = consumers.")
    hbar(machine, rows, "sensors", "Sensors that wait between readings: where producer-consumer shines",
         "Seconds (lower is better); speed-up vs sequential in brackets", "sensors.png",
         "2,300 readings, each sensor waits 1 ms per reading.")
    print("charts written")
