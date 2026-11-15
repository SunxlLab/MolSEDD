"""ZINC-250k 性质定向条件生成（论文主方法：SEDD 先验 + PUCT MCTS）。

搜索逐 token 生长 SMILES 前缀。每个节点把当前前缀钉住（投影），让 SEDD 模型
采样一批分子，一次调用同时拿到两样东西：

  * reward ``R = exp(-MAE)``：这批分子的 Crippen logP 与目标值的偏差；
  * 下一个 token 的经验分布 ``P(s,a)``：只从这批分子实际产生过的 token 里选动作
    （dynamic vocabulary），并用模型自己的频率作 PUCT 先验。

选择规则::

    Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + n(s,a))

用法::

    python mcts/run_zinc.py --model_path exp_local/zinc/2026.01.01/120000 --wanti 1.0

ZINC 的搜索超参写在下方的 ``DATASET_CONFIG`` 里，都可以用命令行参数覆盖。
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.search import MCTSSearch, SearchConfig, log_print  # noqa: E402
from sedd_mol.generator import MolecularGenerator  # noqa: E402
from sedd_mol.vocab import ZINC_TOKENS, DatasetSpec  # noqa: E402

# --------------------------------------------------------------------------- #
# ZINC 配置（与 configs/config_zinc.yaml 保持一致）
# --------------------------------------------------------------------------- #
SEQ_LEN = 48
SPEC = DatasetSpec(
    name="zinc",
    tokens=ZINC_TOKENS,
    seq_len=SEQ_LEN,
    min_len=29,
    max_len=48,
    start_tokens=["C", "c", "N", "O"],
)

# 论文里 ZINC 用的搜索超参；ablation/run_ablation.py 直接复用这份。
DATASET_CONFIG = dict(
    name="zinc",
    spec=SPEC,
    seq_len=SEQ_LEN,
    max_steps=SEQ_LEN - 1,
    start_tokens=list(SPEC.start_tokens),
    simulations=500,
    c_param=0.618,
    discovery_bs=200,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ZINC-250k property-targeted MCTS (PUCT + SEDD prior)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--wanti", type=float, default=1.0, help="目标 Crippen logP")
    parser.add_argument("--simulations", type=int, default=DATASET_CONFIG["simulations"])
    parser.add_argument("--c_param", type=float, default=DATASET_CONFIG["c_param"])
    parser.add_argument("--discovery_bs", type=int, default=DATASET_CONFIG["discovery_bs"],
                        help="每节点采样多少分子，同时用来估 reward 和下一 token 分布")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="每个节点重复采样几次再取平均（降方差）")
    parser.add_argument("--start_tokens", type=str, default=None,
                        help="逗号分隔的首层 token（默认 C,c,N,O）")
    parser.add_argument("--max_steps", type=int, default=DATASET_CONFIG["max_steps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="mcts_out")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_tokens = (
        [t.strip() for t in args.start_tokens.split(",") if t.strip()]
        if args.start_tokens
        else list(DATASET_CONFIG["start_tokens"])
    )

    config = SearchConfig(
        selection="puct",          # 主方法：PUCT
        prior="sedd",              # 先验来自 SEDD 自己的下一 token 分布
        dynamic_vocab=True,        # 动作空间由模型采样动态给出
        validity_filter=True,      # 丢弃解析不了的分子的动作
        simulations=args.simulations,
        c_param=args.c_param,
        discovery_bs=args.discovery_bs,
        eval_bs=args.discovery_bs,
        num_repeats=args.num_repeats,
        target=args.wanti,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    log_print(f"[ZINC] 首层 token: {start_tokens}")
    generator = MolecularGenerator(
        args.model_path, seq_len=SEQ_LEN, device=args.device, spec=SPEC
    )
    search = MCTSSearch(generator, SPEC, config)

    start = time.time()
    results = search.run(start_tokens=start_tokens)
    elapsed = time.time() - start

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"zinc_puct_logp{args.wanti}_{time.strftime('%Y%m%d_%H%M%S')}"
    search.save_csv(os.path.join(args.output_dir, f"{tag}.csv"))
    with open(os.path.join(args.output_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"args": vars(args), "elapsed_sec": elapsed, "results": results},
            f,
            indent=2,
            default=str,
        )
    log_print(f"[ZINC] 用时 {elapsed / 60:.1f} min，结果写入 {args.output_dir}/{tag}.json")


if __name__ == "__main__":
    main()
