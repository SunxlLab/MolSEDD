"""MCTS-based conditional generation (all variants of the paper).

Variants (see ``VARIANTS``)::

    puct                  proposed method: PUCT + SEDD prior + dynamic vocabulary + validity filter
    uct                   UCT + dynamic vocabulary + validity filter
    uct_novalid           UCT + dynamic vocabulary, no validity filter
    pure                  pure MCTS: UCT + static (full) vocabulary
    puct_uniform          PUCT with a uniform prior (removes the SEDD prior)
    puct_novalid          PUCT + SEDD prior, no validity filter
    puct_uniform_novalid  PUCT + uniform prior, no validity filter

Example::

    python -m condgen.run_mcts --variant puct --dataset moses \
        --model_path exp_local/moses/2026.02.26/104218 --wanti 1.0 --simulations 500
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from sedd_mol.generator import MolecularGenerator
from sedd_mol.vocab import get_spec

from .mcts import MCTSSearch, SearchConfig, log_print

VARIANTS = {
    "puct": dict(selection="puct", prior="sedd", dynamic_vocab=True, validity_filter=True),
    "uct": dict(selection="uct", prior="uniform", dynamic_vocab=True, validity_filter=True),
    "uct_novalid": dict(selection="uct", prior="uniform", dynamic_vocab=True, validity_filter=False),
    "pure": dict(selection="uct", prior="uniform", dynamic_vocab=False, validity_filter=False),
    "puct_uniform": dict(selection="puct", prior="uniform", dynamic_vocab=True, validity_filter=True),
    "puct_novalid": dict(selection="puct", prior="sedd", dynamic_vocab=True, validity_filter=False),
    "puct_uniform_novalid": dict(selection="puct", prior="uniform", dynamic_vocab=True, validity_filter=False),
}


def build_parser():
    parser = argparse.ArgumentParser(description="Property-targeted MCTS over SMILES prefixes")
    parser.add_argument("--variant", type=str, default="puct", choices=sorted(VARIANTS))
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="moses", choices=["qm9", "zinc", "moses"])
    parser.add_argument("--wanti", type=float, default=1.0, help="Target Crippen logP")
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--c_param", type=float, default=0.618)
    parser.add_argument("--discovery_bs", type=int, default=100,
                        help="Batch size used to discover children and estimate the reward")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Alias of --discovery_bs (kept for the original scripts)")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Independent batches averaged per simulation (variance reduction)")
    parser.add_argument("--start_tokens", type=str, default=None,
                        help="Comma separated first-layer tokens (default: dataset default)")
    parser.add_argument("--max_steps", type=int, default=None, help="Default: seq_len - 1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="condgen_out")
    # explicit overrides of the variant
    parser.add_argument("--selection", type=str, default=None, choices=["puct", "uct"])
    parser.add_argument("--prior", type=str, default=None, choices=["sedd", "uniform"])
    parser.add_argument("--vocab", type=str, default=None, choices=["dynamic", "static"])
    parser.add_argument("--validity_filter", type=str, default=None, choices=["on", "off"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    spec = get_spec(args.dataset)
    preset = dict(VARIANTS[args.variant])
    if args.selection:
        preset["selection"] = args.selection
    if args.prior:
        preset["prior"] = args.prior
    if args.vocab:
        preset["dynamic_vocab"] = args.vocab == "dynamic"
    if args.validity_filter:
        preset["validity_filter"] = args.validity_filter == "on"

    discovery_bs = args.batch_size or args.discovery_bs
    start_tokens = (
        [t.strip() for t in args.start_tokens.split(",") if t.strip()]
        if args.start_tokens
        else list(spec.start_tokens)
    )

    config = SearchConfig(
        simulations=args.simulations,
        c_param=args.c_param,
        discovery_bs=discovery_bs,
        eval_bs=discovery_bs,
        num_repeats=args.num_repeats,
        target=args.wanti,
        max_steps=args.max_steps or (spec.seq_len - 1),
        seed=args.seed,
        **preset,
    )

    log_print(f"Initial (root) tokens: {start_tokens}")
    generator = MolecularGenerator(args.model_path, dataset=args.dataset)
    search = MCTSSearch(generator, spec, config)

    start = time.time()
    results = search.run(start_tokens=start_tokens)
    elapsed = time.time() - start

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.variant}_logp{args.wanti}_{time.strftime('%Y%m%d_%H%M%S')}"
    search.save_csv(os.path.join(args.output_dir, f"{tag}.csv"))
    with open(os.path.join(args.output_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"args": vars(args), "variant": preset, "elapsed_sec": elapsed, "results": results},
            f,
            indent=2,
            default=str,
        )
    log_print(f"Done in {elapsed / 60:.1f} min. Results written to {args.output_dir}/{tag}.json")


if __name__ == "__main__":
    main()
