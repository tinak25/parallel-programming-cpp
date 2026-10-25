#!/usr/bin/env python3
"""Generates a traffic data file: one reading per light every 5 minutes.

Car counts follow a daily pattern (quiet at night, peaks around 8:00 and
17:00), each light has its own base level, and a few malformed lines are
mixed in so the parsers' error handling is exercised.

    python3 tests/generate_data.py --days 1 --lights 8 --out data/sample.csv
"""
import argparse
import datetime as dt
import math
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--lights", type=int, default=8)
    p.add_argument("--start", default="2025-03-01 16:00")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    base = {f"TL{i:02d}": rng.uniform(40, 160) for i in range(1, args.lights + 1)}
    t = dt.datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    end = t + dt.timedelta(days=args.days)
    with open(args.out, "w") as f:
        f.write("# timestamp,light_id,cars\n")
        while t < end:
            h = t.hour + t.minute / 60
            daily = 0.25 + math.exp(-((h - 8) ** 2) / 4) + 1.2 * math.exp(-((h - 17) ** 2) / 5)
            for light, level in base.items():
                cars = max(0, round(level * daily * rng.uniform(0.8, 1.2)))
                f.write(f"{t:%Y-%m-%d %H:%M},{light},{cars}\n")
            if rng.random() < 0.01:
                f.write(rng.choice(["", "bad line", f"{t:%Y-%m-%d %H:%M},TL01,lots", "25:99,TL01,10"]) + "\n")
            t += dt.timedelta(minutes=5)


if __name__ == "__main__":
    main()
