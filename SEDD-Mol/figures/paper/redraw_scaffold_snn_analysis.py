"""
Nature 投稿风格：Scaffold 分析 + 最近邻相似度（SNN）分析。

验证目的: 证明条件生成（树搜索）没有"背训练集"，其分子既新颖又多样。
    - 实验1 Scaffold Novelty      : 生成骨架中未出现在训练集的比例
    - 实验1 Unique Scaffold Ratio : unique scaffolds / unique valid molecules
    - 实验2 SNN (nearest-neighbor similarity) : 每个生成分子与训练集最相似分子的最大 Tanimoto

数据:
    训练集   train/trainMOSES.txt   157.8 万分子 -> 全量(构建 scaffold 集合 + 指纹库)
    无条件   generate/MOSES.txt     11887 -> 抽 1000 有效分子
    条件     cond/MOSESto1.txt      5120  -> 抽 1000 有效分子 (目标 LogP=1)

输出:
    数字 -> 终端 + 图 (Nature 风格, 两个 panel: a=SNN 分布, b=Scaffold 指标柱状图)

用法:
    python redraw_scaffold_snn_analysis.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

NATUREIMG_DIR = Path(__file__).resolve().parents[1]
TODOIMAGE_DIR = NATUREIMG_DIR.parent
OUTPUT_DIR = NATUREIMG_DIR / "images"

EXPORT_FORMATS = ["pdf", "svg", "eps", "tiff", "jpg"]
FORMAT_SUBDIRS = {fmt: fmt for fmt in EXPORT_FORMATS}
DPI = 600

TEXT_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
UNCOND_COLOR = "#666666"
COND_COLOR = "#08519C"

GEN_SAMPLE = 1000       # 每组生成分子抽样数
N_BITS = 2048           # Morgan fingerprint 位数
RADIUS = 2
SEED = 42


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> None:
    ensure_output_dir(output_dir)
    common = {"bbox_inches": "tight", "facecolor": "white"}
    for fmt in formats:
        format_dir = output_dir / FORMAT_SUBDIRS.get(fmt, fmt)
        ensure_output_dir(format_dir)
        if fmt == "pdf":
            fig.savefig(format_dir / f"{stem}.pdf", format="pdf", **common)
        elif fmt == "svg":
            fig.savefig(format_dir / f"{stem}.svg", format="svg", **common)
        elif fmt == "eps":
            fig.savefig(format_dir / f"{stem}.eps", format="eps", dpi=DPI, **common)
        elif fmt == "tiff":
            fig.savefig(format_dir / f"{stem}.tiff", format="tiff", dpi=DPI, **common)
        elif fmt == "jpg":
            fig.savefig(format_dir / f"{stem}.jpg", format="jpg", dpi=DPI, pil_kwargs={"quality": 95}, **common)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, **common)
    plt.close(fig)


def read_smiles(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def deterministic_sample(items: Sequence[str], limit: int, seed_offset: int = 0) -> List[str]:
    candidates = list(items)
    if len(candidates) > 1:
        rng = random.Random(SEED + seed_offset)
        rng.shuffle(candidates)
    return candidates[:limit]


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def scaffold_smiles(mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def process_molecules(smiles_list: Sequence[str], limit: int):
    """解析有效分子, 返回 (有效SMILES, scaffold列表, 指纹列表)"""
    valid_smiles: List[str] = []
    scaffolds: List[str] = []
    fps = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        valid_smiles.append(smi)
        scaffolds.append(scaffold_smiles(mol))
        fps.append(morgan_fp(mol))
        if len(valid_smiles) >= limit:
            break
    return valid_smiles, scaffolds, fps


def extract_scaffolds_and_fps_all(smiles_list: Sequence[str]):
    """全量解析训练集，一次遍历同时提取 scaffold + 指纹，返回 (scaffold列表, 指纹列表, 有效分子数)"""
    scaffolds: List[str] = []
    fps = []
    valid_count = 0
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        valid_count += 1
        scaffolds.append(scaffold_smiles(mol))
        fps.append(morgan_fp(mol))
    return scaffolds, fps, valid_count


def fps_to_numpy(fps) -> np.ndarray:
    arr = np.zeros((len(fps), N_BITS), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def compute_snn(query_np: np.ndarray, train_np: np.ndarray, block_size: int = 10000) -> np.ndarray:
    """向量化计算每个查询分子与训练集的最大 Tanimoto (SNN)"""
    q = query_np.astype(np.float32)
    q_sum = q.sum(axis=1, dtype=np.float32)
    n_train = train_np.shape[0]
    snn = np.zeros(q.shape[0], dtype=np.float32)
    for start in range(0, n_train, block_size):
        end = min(start + block_size, n_train)
        t = train_np[start:end].astype(np.float32)
        inter = q @ t.T
        t_sum = t.sum(axis=1, dtype=np.float32)
        union = q_sum[:, None] + t_sum[None, :] - inter
        tanimoto = inter / np.maximum(union, 1e-9)
        snn = np.maximum(snn, tanimoto.max(axis=1))
    return snn


def scaffold_stats(scaffolds: Sequence[str], valid_smiles: Sequence[str], train_scaffold_set: set) -> Tuple[float, float]:
    unique_scaffolds = set(scaffolds)
    unique_mols = set(valid_smiles)
    if not unique_scaffolds or not unique_mols:
        return 0.0, 0.0
    novel = sum(1 for s in unique_scaffolds if s not in train_scaffold_set) / len(unique_scaffolds)
    ratio = len(unique_scaffolds) / len(unique_mols)
    return float(novel), float(ratio)


def main() -> None:
    set_publication_style()

    train_path = TODOIMAGE_DIR / "train" / "trainMOSES.txt"
    uncond_path = TODOIMAGE_DIR / "generate" / "MOSES.txt"
    cond_path = TODOIMAGE_DIR / "cond" / "MOSESto1.txt"

    print("[1/5] 读取数据...")
    train_all_smiles = read_smiles(train_path)  # 全量 157 万
    uncond_sample = deterministic_sample(read_smiles(uncond_path), GEN_SAMPLE, seed_offset=1)
    cond_sample = deterministic_sample(read_smiles(cond_path), GEN_SAMPLE, seed_offset=2)

    print(f"[2/5] 全量解析训练集 (scaffold + 指纹, 共 {len(train_all_smiles)} 个分子)...")
    train_all_scaffolds, train_all_fps, train_valid_n = extract_scaffolds_and_fps_all(train_all_smiles)
    train_scaffold_set = set(train_all_scaffolds)

    print("[3/5] 处理生成分子...")
    uncond_mols, uncond_scaffolds, uncond_fps = process_molecules(uncond_sample, GEN_SAMPLE)
    cond_mols, cond_scaffolds, cond_fps = process_molecules(cond_sample, GEN_SAMPLE)

    print("[4/5] 计算指标...")
    # 训练集自身的 unique scaffold ratio（全量参考值）
    train_ratio = len(train_scaffold_set) / train_valid_n if train_valid_n else 0.0

    uncond_novelty, uncond_ratio = scaffold_stats(uncond_scaffolds, uncond_mols, train_scaffold_set)
    cond_novelty, cond_ratio = scaffold_stats(cond_scaffolds, cond_mols, train_scaffold_set)

    # SNN（全量指纹库）
    train_np = fps_to_numpy(train_all_fps)
    uncond_snn = compute_snn(fps_to_numpy(uncond_fps), train_np)
    cond_snn = compute_snn(fps_to_numpy(cond_fps), train_np)

    print("\n" + "=" * 64)
    print("指标结果")
    print("=" * 64)
    print(f"训练集有效分子(全量): {train_valid_n}  唯一骨架数: {len(train_scaffold_set)}")
    print(f"训练集 Unique Scaffold Ratio: {train_ratio:.3f}")
    print("-" * 64)
    print(f"{'指标':<28}{'无条件生成':>16}{'条件生成':>16}")
    print("-" * 64)
    print(f"{'Scaffold Novelty':<28}{uncond_novelty*100:>15.1f}%{cond_novelty*100:>15.1f}%")
    print(f"{'Unique Scaffold Ratio':<28}{uncond_ratio:>16.3f}{cond_ratio:>16.3f}")
    print(f"{'SNN 均值':<28}{np.mean(uncond_snn):>16.3f}{np.mean(cond_snn):>16.3f}")
    print(f"{'SNN 中位数':<28}{np.median(uncond_snn):>16.3f}{np.median(cond_snn):>16.3f}")
    print(f"{'SNN 最大值':<28}{np.max(uncond_snn):>16.3f}{np.max(cond_snn):>16.3f}")
    print("=" * 64)

    print("\n[5/5] 画图...")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # panel a: SNN 分布
    ax = axes[0]
    x_range = np.linspace(0.0, 1.0, 400)
    for values, color, label, ls in [
        (uncond_snn, UNCOND_COLOR, "Unconditional", "--"),
        (cond_snn, COND_COLOR, "Conditional", "-"),
    ]:
        try:
            kde = gaussian_kde(values)
            y = kde(x_range)
            ax.plot(x_range, y, color=color, linestyle=ls, linewidth=1.4, label=label)
        except Exception:
            pass
    ax.set_xlabel("Nearest-neighbor similarity")
    ax.set_ylabel("Probability density")
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0)
    ax.tick_params(axis="both", pad=1.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8)
    ax.set_title("Nearest-neighbor similarity", fontsize=8)

    # panel b: scaffold 指标柱状图
    ax = axes[1]
    categories = ["Unconditional", "Conditional"]
    x = np.arange(len(categories))
    width = 0.35
    novelty_vals = [uncond_novelty, cond_novelty]
    ratio_vals = [uncond_ratio, cond_ratio]
    ax.bar(x - width / 2, novelty_vals, width, color="#9ECAE1", label="Scaffold novelty", edgecolor="none")
    ax.bar(x + width / 2, ratio_vals, width, color="#08519C", label="Unique scaffold ratio", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0, axis="y")
    ax.tick_params(axis="both", pad=1.5)
    ax.legend(loc="upper right", frameon=False, handlelength=1.2)
    ax.set_title("Scaffold metrics", fontsize=8)

    # 柱子上标数值
    for bar_x, v in zip([-width / 2, width / 2], novelty_vals):
        ax.text(bar_x, v + 0.03, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=6, color="#1F5AA6")
    for bar_x, v in zip([1 - width / 2, 1 + width / 2], ratio_vals):
        ax.text(bar_x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color="#08519C")

    fig.tight_layout()
    stem = "Fig_scaffold_snn_analysis"
    save_figure(fig, OUTPUT_DIR, stem, EXPORT_FORMATS)

    format_dirs = ", ".join(str(OUTPUT_DIR / FORMAT_SUBDIRS[fmt]) for fmt in EXPORT_FORMATS)
    print(f"\n[natureimg] 图片已导出到: {format_dirs}")
    print(f"[natureimg] PNG 预览: {OUTPUT_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()
