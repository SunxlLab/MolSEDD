"""Prefix / suffix constrained generation.

    python sample_cond.py --model_path exp_local/moses/2026.02.26/104218 \
                          --dataset moses --prefix "CC" --num_samples 1000 --wanti 1.0

The prefix is pinned token by token at every denoising step, so every generated
molecule starts with it; the suffix defaults to the end token ``$``.
"""

import argparse
import json
import os
import time

import torch

from sedd_mol.generator import MolecularGenerator
from sedd_mol.metrics import distribution_metrics, format_metrics, property_scores
from sedd_mol.vocab import get_spec


def main():
    parser = argparse.ArgumentParser(description="Conditional sampling (prefix/suffix projection)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Experiment directory produced by train.py, or a HuggingFace repo id")
    parser.add_argument("--dataset", type=str, default="moses", choices=["qm9", "zinc", "moses"])
    parser.add_argument("--prefix", type=str, default="", help="SMILES prefix that must be kept")
    parser.add_argument("--suffix", type=str, default="$", help="SMILES suffix (default: end token)")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--wanti", type=float, default=None,
                        help="Optional target logP used to report avg logP / MAE / reward")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    spec = get_spec(args.dataset)
    gen = MolecularGenerator(
        args.model_path, dataset=args.dataset, seq_len=args.seq_len, device=args.device
    )

    start = time.perf_counter()
    smiles = []
    remaining = args.num_samples
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        smiles.extend(gen.generate(prefix=args.prefix, suffix=args.suffix, batch_size=bs, steps=args.steps))
        remaining -= bs
    elapsed = time.perf_counter() - start

    output = args.output or os.path.join(
        "samples", f"cond_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in smiles)

    print(f"Prefix: {args.prefix!r} | generated {len(smiles)} molecules in "
          f"{elapsed / 60:.2f} min ({len(smiles) / elapsed:.2f} mol/s)")
    print(f"Saved to {output}")
    print(format_metrics(distribution_metrics(smiles)))

    if args.wanti is not None:
        scores = property_scores(smiles, args.wanti)
        print(json.dumps({k: v for k, v in scores.items() if k != "best_smiles"}, indent=2))
        print(f"Best molecule: {scores['best_smiles']}")


if __name__ == "__main__":
    main()
