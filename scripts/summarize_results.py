"""Aggregate evaluate_GReMS.py result JSONs (one per seed) into a per-node-size table.

Usage:
    python scripts/summarize_results.py results/GReMS_Final_Finetune_rgg*.json
    python scripts/summarize_results.py results/*_rgg*.json --csv summary.csv

Each input file is one evaluation seed. For every node count N the script reports the
mean end-to-end latency (ms) of each method averaged over the test networks of that
size, then the mean and standard deviation of that quantity across seeds.
Baseline columns are shown only if the files were produced without --no-baselines.
"""

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict

METHODS = [("GReMS", "rl_mean"), ("DiSNet", "disnet_mean"),
           ("MoDNN", "modnn_mean"), ("EnSplit", "ensplit_mean")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="result JSON files (one per seed)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = ap.parse_args()

    per_size = defaultdict(lambda: defaultdict(list))   # size -> method -> [value per seed]
    for path in args.files:
        with open(path) as f:
            data = json.load(f)
        for size, row in data.get("size_summary", {}).items():
            for name, key in METHODS:
                if row.get(key) is not None:
                    per_size[int(size)][name].append(row[key])

    if not per_size:
        sys.exit("No size_summary found in the given files.")

    methods = [m for m, _ in METHODS if any(m in per_size[s] for s in per_size)]
    n_seeds = len(args.files)
    header = ["N"] + [f"{m} (ms)" for m in methods]
    rows = []
    for size in sorted(per_size):
        row = [size]
        for m in methods:
            vals = per_size[size].get(m, [])
            if not vals:
                row.append("n/a")
            else:
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                row.append(f"{st.mean(vals):.0f} ± {sd:.0f}")
        rows.append(row)

    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(f"Mean latency per node count, mean ± std over {n_seeds} seed file(s)\n")
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
