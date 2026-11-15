"""QM9 前缀条件生成：给定前缀，让模型把剩下的部分补全。

条件生成本质上就是"无条件生成 + 前缀"：每一步去噪后，把前缀和结束符的 token
原样写回序列（投影，见 ``sedd/sampling.py`` 的 ``proj_fun``），模型只能补中间
空出来的位置。不需要额外参数，也不需要微调。

用法::

    python conditional/cond_qm9.py --model_path exp_local/qm9/2026.01.01/120000 \
        --prefix "FOCC1=C" --wanti 1.0

序列末尾固定为结束符。
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd_mol.generator import MolecularGenerator  # noqa: E402
from sedd_mol.metrics import property_scores  # noqa: E402
from sedd_mol.vocab import QM9_TOKENS, DatasetSpec  # noqa: E402

# --------------------------------------------------------------------------- #
# QM9 配置（与 configs/config.yaml 保持一致）
# --------------------------------------------------------------------------- #
SEQ_LEN = 22
SPEC = DatasetSpec(
    name="qm9",
    tokens=QM9_TOKENS,
    seq_len=SEQ_LEN,
    min_len=6,
    max_len=22,
    start_tokens=["C", "N", "O", "F"],
)


def parse_args():
    parser = argparse.ArgumentParser(description="QM9 conditional generation (prefix projection)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="", help="必须保留的 SMILES 前缀")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--wanti", type=float, default=None,
                        help="目标 logP；给了就额外报告 avg logP / MAE / reward")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件（默认 samples/cond_qm9_<时间戳>.txt）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    gen = MolecularGenerator(
        args.model_path, seq_len=args.seq_len, device=args.device, spec=SPEC
    )

    start = time.perf_counter()
    smiles = []
    remaining = args.num_samples
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        smiles.extend(
            gen.generate(prefix=args.prefix, batch_size=batch, steps=args.steps)
        )
        remaining -= batch
    elapsed = time.perf_counter() - start

    output = args.output or os.path.join(
        "samples", f"cond_qm9_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in smiles)

    kept = sum(1 for s in smiles if s.startswith(args.prefix)) if args.prefix else len(smiles)
    print(f"[QM9-cond] 前缀 {args.prefix!r} | 生成 {len(smiles)} 个分子，"
          f"用时 {elapsed / 60:.2f} min ({len(smiles) / elapsed:.2f} mol/s)")
    print(f"[QM9-cond] 前缀保留率: {kept}/{len(smiles)} "
          f"({kept / max(len(smiles), 1):.2%})")
    print(f"[QM9-cond] 已保存到 {output}")

    if args.wanti is not None:
        scores = property_scores(smiles, args.wanti)
        print("[QM9-cond] 目标性质:")
        print(json.dumps({k: v for k, v in scores.items() if k != "best_smiles"}, indent=2))
        print(f"[QM9-cond] 最接近目标的分子: {scores['best_smiles']}")


if __name__ == "__main__":
    main()
