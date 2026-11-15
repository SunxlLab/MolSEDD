"""消融实验：论文 4.5 的四条曲线，由同一个引擎的四个开关组合而成。

四个开关（``SearchConfig``）::

    selection        puct | uct               选择规则
    prior            sedd | uniform           PUCT 的先验来源
    dynamic_vocab    True | False             动作空间由模型动态给出 / 用整个词表
    validity_filter  True | False             丢弃解析不了的分子的动作

四个变体（对应论文图 5 的四条线）::

    puct           PUCT + SEDD 先验 + 动态词表 + 有效性过滤   ← MolSEDD（本文方法）
    uct            UCT  + 动态词表 + 有效性过滤
    uct_novalid    UCT  + 动态词表，去掉有效性过滤
    pure           纯 MCTS：UCT + 完整静态词表

搜索超参复用 ``mcts/run_{qm9,zinc,moses}.py`` 的 ``DATASET_CONFIG``
（500 次模拟、c=0.618），但**每节点补全数固定为 100**，与论文 4.5 和
附录 S1.6 一致（主实验用的是 200）。

用法::

    # MOSES 上跑四条线
    python ablation/run_ablation.py --dataset moses --model_path exp_local/moses/2026.02.26/104218

    # 只跑其中一条
    python ablation/run_ablation.py --dataset moses --model_path ... --variants puct
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
from mcts.run_moses import DATASET_CONFIG as MOSES_CONFIG  # noqa: E402
from mcts.run_qm9 import DATASET_CONFIG as QM9_CONFIG  # noqa: E402
from mcts.run_zinc import DATASET_CONFIG as ZINC_CONFIG  # noqa: E402
from sedd_mol.generator import MolecularGenerator  # noqa: E402

DATASET_CONFIGS = {"qm9": QM9_CONFIG, "zinc": ZINC_CONFIG, "moses": MOSES_CONFIG}

VARIANTS = {
    "puct": dict(selection="puct", prior="sedd", dynamic_vocab=True, validity_filter=True),
    "uct": dict(selection="uct", prior="uniform", dynamic_vocab=True, validity_filter=True),
    "uct_novalid": dict(selection="uct", prior="uniform", dynamic_vocab=True, validity_filter=False),
    "pure": dict(selection="uct", prior="uniform", dynamic_vocab=False, validity_filter=False),
}

ALL_VARIANTS = list(VARIANTS)

# 论文 4.5 / 附录 S1.6 用的每节点补全数。
# 主实验（表 4）用的是 200，见 mcts/run_moses.py。
ABLATION_DISCOVERY_BS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="MCTS 消融实验")
    parser.add_argument("--dataset", type=str, default="moses", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--variants", type=str, default="all",
                        help="逗号分隔的变体名，或 'all'（默认四条线全跑）")
    parser.add_argument("--wanti", type=float, default=1.0, help="目标 Crippen logP")
    parser.add_argument("--simulations", type=int, default=None,
                        help="覆盖该数据集默认的搜索次数")
    parser.add_argument("--c_param", type=float, default=None)
    parser.add_argument("--discovery_bs", type=int, default=ABLATION_DISCOVERY_BS,
                        help="每节点补全数，论文 4.5 用的是 100")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 mcts_out/ablation")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_variant(name, dataset_cfg, args, generator, output_dir):
    preset = VARIANTS[name]
    config = SearchConfig(
        simulations=args.simulations or dataset_cfg["simulations"],
        c_param=args.c_param if args.c_param is not None else dataset_cfg["c_param"],
        discovery_bs=args.discovery_bs,
        eval_bs=args.discovery_bs,
        num_repeats=args.num_repeats,
        target=args.wanti,
        max_steps=dataset_cfg["max_steps"],
        seed=args.seed,
        **preset,
    )

    log_print(
        f"\n===== {dataset_cfg['name']} / {name} ====="
        f" selection={preset['selection']} prior={preset['prior']}"
        f" dynamic_vocab={preset['dynamic_vocab']}"
        f" validity_filter={preset['validity_filter']}"
        f" simulations={config.simulations} c_param={config.c_param}"
    )

    search = MCTSSearch(generator, dataset_cfg["spec"], config)
    start = time.time()
    results = search.run(start_tokens=dataset_cfg["start_tokens"])
    elapsed = time.time() - start

    tag = f"{dataset_cfg['name']}_{name}_logp{args.wanti}"
    search.save_csv(os.path.join(output_dir, f"{tag}.csv"))
    with open(os.path.join(output_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"variant": preset, "elapsed_sec": elapsed, "results": results},
            f,
            indent=2,
            default=str,
        )
    log_print(f"[{dataset_cfg['name']}/{name}] 用时 {elapsed / 60:.1f} min -> {tag}.json")
    return tag


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    variants = ALL_VARIANTS if args.variants == "all" else [
        v.strip() for v in args.variants.split(",") if v.strip()
    ]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"未知变体 {unknown}，可选：{ALL_VARIANTS}")

    output_dir = args.output_dir or os.path.join("mcts_out", "ablation")
    os.makedirs(output_dir, exist_ok=True)

    # 三个数据集共享同一个模型和 generator，只换搜索配置。
    generator = MolecularGenerator(
        args.model_path,
        seq_len=dataset_cfg["seq_len"],
        device=args.device,
        spec=dataset_cfg["spec"],
    )

    tags = []
    for name in variants:
        tags.append(run_variant(name, dataset_cfg, args, generator, output_dir))

    log_print(f"\n全部完成：{len(tags)} 个变体，结果在 {output_dir}/")
    for tag in tags:
        log_print(f"  {tag}.csv")


if __name__ == "__main__":
    main()
