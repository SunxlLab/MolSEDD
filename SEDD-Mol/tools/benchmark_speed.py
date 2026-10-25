"""Sampling-throughput benchmark over batch sizes (data for the speed figure).

    python tools/benchmark_speed.py --model_path exp_local/moses/2026.02.26/104218 \
        --dataset moses --batches 1,10,100,500,1000,2000 --steps 1024 \
        --output results/speed_sweep.csv
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.load_model import load_model  # noqa: E402
from sedd.sampling import get_pc_sampler  # noqa: E402
from sedd_mol.metrics import is_valid_smiles  # noqa: E402
from sedd_mol.vocab import get_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Measure molecules/second for several batch sizes")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="moses", choices=["qm9", "zinc", "moses"])
    parser.add_argument("--batches", type=str, default="1,10,100,500,1000,2000")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/speed_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_spec(args.dataset)
    model, graph, noise = load_model(args.model_path, device)
    model.eval()

    rows = []
    for bs in [int(b) for b in args.batches.split(",")]:
        sampler = get_pc_sampler(graph, noise, (bs, spec.seq_len), "analytic", args.steps, device=device)
        for _ in range(args.warmup):
            sampler(model)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            samples = sampler(model)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

        elapsed = min(timings)
        smiles = [spec.decode(r.tolist()) for r in samples.cpu().numpy()]
        valid = sum(is_valid_smiles(s) for s in smiles) / len(smiles)
        row = {
            "batch_size": bs,
            "seconds": round(elapsed, 3),
            "samples_per_sec": round(bs / elapsed, 3),
            "ms_per_molecule": round(elapsed / bs * 1000, 1),
            "validity": round(valid, 4),
        }
        rows.append(row)
        print(row, flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
