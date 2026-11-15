"""MOSES 无条件生成：加载训练好的 SEDD 模型采样 SMILES。

用法::

    python generate/sample_moses.py --model_path exp_local/moses/2026.01.01/120000

序列长度和词表写在下方的 ``SPEC`` 里，必须与训练该 checkpoint 时一致。
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd.sampling import get_pc_sampler  # noqa: E402
from sedd_mol.vocab import MOSES_TOKENS, DatasetSpec  # noqa: E402

# --------------------------------------------------------------------------- #
# MOSES 配置（与 configs/config_moses.yaml 保持一致）
# --------------------------------------------------------------------------- #
SEQ_LEN = 47
SPEC = DatasetSpec(
    name="moses",
    tokens=MOSES_TOKENS,
    seq_len=SEQ_LEN,
    min_len=24,
    max_len=47,
    start_tokens=["Br", "C", "Cl", "F", "N", "O", "[H]", "c"],
)


def parse_args():
    parser = argparse.ArgumentParser(description="MOSES unconditional generation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="train.py 产出的实验目录，或 HuggingFace repo id")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1024, help="扩散步数")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="覆盖模型上下文长度")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件（默认 samples/moses_<时间戳>.txt）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from sedd.load_model import load_model

    model, graph, noise = load_model(args.model_path, device)
    model.eval()

    start = time.perf_counter()
    smiles = []
    remaining = args.num_samples
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        sampling_fn = get_pc_sampler(
            graph, noise, (batch, args.seq_len), "analytic", args.steps, device=device
        )
        ids = sampling_fn(model).cpu().numpy()
        for row in ids:
            smiles.append(SPEC.decode(row.tolist()))
        remaining -= batch
    elapsed = time.perf_counter() - start

    output = args.output or os.path.join(
        "samples", f"moses_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in smiles)

    print(f"[MOSES] 生成 {len(smiles)} 个分子，用时 {elapsed / 60:.2f} min "
          f"({len(smiles) / elapsed:.2f} mol/s)")
    print(f"[MOSES] 已保存到 {output}")


if __name__ == "__main__":
    main()
