"""Validity / uniqueness / novelty of a generated SMILES file.

    python tools/eval_metrics.py samples/moses_20260101_120000.txt \
        --train data/moses_train.txt --sample 1000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd_mol.dataset import load_smiles  # noqa: E402
from sedd_mol.metrics import distribution_metrics, format_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Distribution-level molecule metrics")
    parser.add_argument("generated", type=str, help="File with one generated SMILES per line")
    parser.add_argument("--train", type=str, default=None, help="Training SMILES file (for novelty)")
    parser.add_argument("--sample", type=int, default=1000, help="Number of molecules to score")
    parser.add_argument("--no_filter", action="store_true", help="Disable the syntax pre-filter")
    parser.add_argument("--json", action="store_true", help="Print the metrics as JSON")
    args = parser.parse_args()

    generated = load_smiles(args.generated)
    train = load_smiles(args.train) if args.train else None
    metrics = distribution_metrics(
        generated, train, sample_size=args.sample, apply_filter=not args.no_filter
    )
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
