"""Training / evaluation loss curves from a ``train.py`` log file.

    python figures/plot_loss.py --log exp_local/moses/2026.02.26/104218/logs --output figures/loss.pdf
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_RE = re.compile(r"step:\s*(\d+),\s*training_loss:\s*([0-9.eE+-]+)")
EVAL_RE = re.compile(r"step:\s*(\d+),\s*evaluation_loss:\s*([0-9.eE+-]+)")


def main():
    parser = argparse.ArgumentParser(description="Plot training and evaluation loss")
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/loss.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    train, evals = [], []
    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if m := TRAIN_RE.search(line):
                train.append((int(m.group(1)), float(m.group(2))))
            elif m := EVAL_RE.search(line):
                evals.append((int(m.group(1)), float(m.group(2))))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if train:
        ax.plot([s for s, _ in train], [l for _, l in train], linewidth=0.8, label="train")
    if evals:
        ax.plot([s for s, _ in evals], [l for _, l in evals], "o-", markersize=3, label="eval")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output} ({len(train)} train points, {len(evals)} eval points)")


if __name__ == "__main__":
    main()
