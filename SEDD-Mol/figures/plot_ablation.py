"""Best-so-far reward curves for the MCTS ablations (paper: four-line figure).

Each curve comes from a ``*/run_mcts.py`` trajectory CSV (``condgen_out/*.csv``)::

    python figures/plot_ablation.py \
        --run "SEDD-prior MCTS (ours)=condgen_out/moses_puct_logp1.0_*.csv" \
        --run "UCT + dynamic vocab=condgen_out/moses_uct_logp1.0_*.csv" \
        --run "UCT, no validity filter=condgen_out/moses_uct_novalid_logp1.0_*.csv" \
        --run "Pure MCTS=condgen_out/moses_pure_logp1.0_*.csv" \
        --output figures/ablation_best_so_far.pdf
"""

import argparse
import csv
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_curve(path):
    rewards, best, out = [], 0.0, []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            best = max(best, rewards[-1])
            out.append(best)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot best-so-far reward curves")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=GLOB",
                        help="Repeatable. Label and a glob for the trajectory CSV")
    parser.add_argument("--output", type=str, default="figures/ablation_best_so_far.pdf")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    for spec in args.run:
        label, pattern = spec.split("=", 1)
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No CSV matched {pattern!r}")
        curves = [read_curve(m) for m in matches]
        n = min(len(c) for c in curves)
        mean = [sum(c[i] for c in curves) / len(curves) for i in range(n)]
        ax.plot(range(1, n + 1), mean, label=label, linewidth=1.6)

    ax.set_xlabel("MCTS simulation")
    ax.set_ylabel("Best-so-far reward  $R=\\exp(-\\mathrm{MAE})$")
    if args.title:
        ax.set_title(args.title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
