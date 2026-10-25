"""Wall-clock sampling cost versus batch size (log-log, with the ideal-linear reference).

    python tools/benchmark_speed.py --model_path ... --output results/speed_sweep.csv
    python figures/plot_speed.py --csv results/speed_sweep.csv --output figures/speed.pdf
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot sampling time vs. batch size")
    parser.add_argument("--csv", type=str, default="results/speed_sweep.csv")
    parser.add_argument("--output", type=str, default="figures/speed_linearity.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    batches, per_mol = [], []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            batches.append(int(row["batch_size"]))
            per_mol.append(float(row["ms_per_molecule"]))

    # ideal linear scaling anchored at the smallest batch
    ideal = [per_mol[0] * (batches[0] / b) for b in batches]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog(batches, per_mol, "o-", label="measured", linewidth=1.6)
    ax.loglog(batches, ideal, "--", color="gray", label="ideal linear")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time per molecule (ms)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
