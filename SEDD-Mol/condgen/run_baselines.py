"""Classical search baselines for property-targeted conditional generation.

    greedy     grow one prefix, each step keeping the token with the best reward
    beam       beam search with configurable width
    beam_mcts  beam search for a few steps, then MCTS seeded with the best beam

All baselines use the same reward as the MCTS search (``R = exp(-MAE)`` over a
batch sampled with the prefix clamped), so numbers are directly comparable.

    python -m condgen.run_baselines --method beam --dataset qm9 \
        --model_path exp_local/qm9/2026.04.06/173406 --wanti 1.0
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from sedd_mol.generator import MolecularGenerator
from sedd_mol.metrics import property_scores
from sedd_mol.vocab import get_spec

from .mcts import MCTSSearch, SearchConfig, log_print


def score_prefix(gen, tokens, target, batch_size, min_validity=0.5):
    smiles = gen.generate(prefix="".join(tokens), batch_size=batch_size)
    return property_scores(smiles, target, min_validity)


def greedy(gen, spec, args, candidates):
    tokens, rows, best_so_far, best_smi, best_logp = [], [], -1.0, None, -100.0
    for step in range(args.max_steps):
        best = None
        for tok in candidates:
            scores = score_prefix(gen, tokens + [tok], args.wanti, args.batch_size)
            if best is None or scores["reward"] > best[1]["reward"]:
                best = (tok, scores)
        tok, scores = best
        tokens.append(tok)
        best_so_far = max(best_so_far, scores["reward"])
        if scores["best_smiles"] and abs(args.wanti - scores["best_logp"]) < abs(args.wanti - best_logp):
            best_smi, best_logp = scores["best_smiles"], scores["best_logp"]
        rows.append({"step": step + 1, "prefix": "".join(tokens), "reward": scores["reward"],
                     "avg_logp": scores["avg_logp"], "mae": scores["mae"],
                     "best_so_far": best_so_far})
        log_print(f"  [Greedy {step + 1}/{args.max_steps}] {''.join(tokens)} | "
                  f"R: {scores['reward']:.4f} | AvgLogP: {scores['avg_logp']:.4f}")
        if scores["reward"] == 0.0:
            log_print("  Prefix died, stopping.")
            break
    return tokens, rows, best_smi, best_logp


def beam_search(gen, spec, args, candidates, depth=None):
    depth = depth or args.max_steps
    beams, rows, best_smi, best_logp = [([], 0.0)], [], None, -100.0
    for step in range(depth):
        expanded = []
        for tokens, _ in beams:
            for tok in candidates:
                scores = score_prefix(gen, tokens + [tok], args.wanti, args.batch_size)
                expanded.append((tokens + [tok], scores["reward"], scores))
                if scores["best_smiles"] and abs(args.wanti - scores["best_logp"]) < abs(args.wanti - best_logp):
                    best_smi, best_logp = scores["best_smiles"], scores["best_logp"]
        expanded.sort(key=lambda x: x[1], reverse=True)
        beams = [(t, r) for t, r, _ in expanded[: args.beam_width]]
        rows.append({"step": step + 1, "prefix": "".join(beams[0][0]),
                     "reward": beams[0][1], "best_so_far": beams[0][1]})
        log_print(f"  [Beam {step + 1}/{depth}] best={''.join(beams[0][0])} R: {beams[0][1]:.4f}")
        if beams[0][1] == 0.0:
            break
    return beams[0][0], rows, best_smi, best_logp


def beam_mcts(gen, spec, args, candidates):
    prefix, rows, _, _ = beam_search(gen, spec, args, candidates, depth=args.beam_depth)
    log_print(f"Beam stage finished, seeding MCTS with {''.join(prefix)!r}")
    config = SearchConfig(
        selection="puct", prior="sedd", dynamic_vocab=True, validity_filter=True,
        simulations=args.simulations, c_param=args.c_param, discovery_bs=args.batch_size,
        eval_bs=args.batch_size, target=args.wanti, max_steps=args.max_steps, seed=args.seed,
    )
    search = MCTSSearch(gen, spec, config)
    results = search.run(root_prefix=prefix)
    return results, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Greedy / beam / beam+MCTS baselines")
    parser.add_argument("--method", type=str, default="beam", choices=["greedy", "beam", "beam_mcts"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "zinc", "moses"])
    parser.add_argument("--wanti", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--beam_depth", type=int, default=3, help="Beam stage depth for beam_mcts")
    parser.add_argument("--simulations", type=int, default=500, help="MCTS budget for beam_mcts")
    parser.add_argument("--c_param", type=float, default=0.618)
    parser.add_argument("--start_tokens", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="condgen_out")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = get_spec(args.dataset)
    args.max_steps = args.max_steps or (spec.seq_len - 1)
    candidates = (
        [t.strip() for t in args.start_tokens.split(",") if t.strip()]
        if args.start_tokens
        else list(spec.tokens)
    )

    gen = MolecularGenerator(args.model_path, dataset=args.dataset)
    start = time.time()
    if args.method == "greedy":
        tokens, rows, best_smi, best_logp = greedy(gen, spec, args, candidates)
        results = {"prefix": "".join(tokens), "best_smiles": best_smi, "best_logp": best_logp}
    elif args.method == "beam":
        tokens, rows, best_smi, best_logp = beam_search(gen, spec, args, candidates)
        results = {"prefix": "".join(tokens), "best_smiles": best_smi, "best_logp": best_logp}
    else:
        results, rows = beam_mcts(gen, spec, args, candidates)

    elapsed = time.time() - start
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.method}_logp{args.wanti}_{time.strftime('%Y%m%d_%H%M%S')}"
    if rows:
        with open(os.path.join(args.output_dir, f"{tag}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.join(args.output_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "elapsed_sec": elapsed, "results": results}, f,
                  indent=2, default=str)
    log_print(f"Done in {elapsed / 60:.1f} min -> {args.output_dir}/{tag}.json")


if __name__ == "__main__":
    main()
