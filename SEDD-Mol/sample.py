"""Unconditional molecule generation.

    python sample.py --model_path exp_local/moses/2026.02.26/104218 \
                     --dataset moses --num_samples 1000 --steps 1024
"""

import argparse
import os
import time

import torch

from sedd.sampling import get_pc_sampler
from sedd_mol.metrics import distribution_metrics, format_metrics, is_valid_smiles
from sedd_mol.vocab import get_spec


def main():
    parser = argparse.ArgumentParser(description="Unconditional sampling from a trained SEDD model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Experiment directory produced by train.py, or a HuggingFace repo id")
    parser.add_argument("--dataset", type=str, default="moses", choices=["qm9", "zinc", "moses"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1024, help="Number of diffusion steps")
    parser.add_argument("--seq_len", type=int, default=None, help="Override the dataset sequence length")
    parser.add_argument("--output", type=str, default=None, help="Where to write the SMILES (default: samples/<dataset>_<time>.txt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    spec = get_spec(args.dataset)
    seq_len = args.seq_len or spec.seq_len
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from sedd.load_model import load_model

    model, graph, noise = load_model(args.model_path, device)
    model.eval()

    start = time.perf_counter()
    smiles, valid = [], 0
    remaining = args.num_samples
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        sampling_fn = get_pc_sampler(graph, noise, (bs, seq_len), "analytic", args.steps, device=device)
        ids = sampling_fn(model).cpu().numpy()
        for row in ids:
            smi = spec.decode(row.tolist())
            smiles.append(smi)
            valid += int(is_valid_smiles(smi))
        remaining -= bs

    output = args.output or os.path.join(
        "samples", f"{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in smiles)

    elapsed = time.perf_counter() - start
    print(f"Generated {len(smiles)} molecules in {elapsed / 60:.2f} min ({len(smiles) / elapsed:.2f} mol/s)")
    print(f"Valid SMILES: {valid}/{len(smiles)} ({valid / max(len(smiles), 1):.2%})")
    print(f"Saved to {output}")
    print(format_metrics(distribution_metrics(smiles)))


if __name__ == "__main__":
    main()
