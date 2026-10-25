"""
Nature 投稿风格：无条件生成 vs 条件生成（树搜索）的 LogP 分布与命中率对比。

论证逻辑（命中率对照实验）:
    固定采样预算 N=1000，对比两种策略落入 |LogP(x) - y*| < eps 的分子占比:
        - Unconditional generation + screening: 无条件生成 1000 个，再筛选
        - Conditional generation (tree search):   树搜索前缀生成 1000 个（目标 y*=1.0）
    用命中率数字证明"无条件生成+筛选"的低效 vs "树搜索"的集中。

图布局:
    左侧: 两条 LogP 的核密度曲线（灰色虚线=无条件, 深蓝实线=条件, 浅蓝填充）
          灰色阴影带标记命中区间 [y*-eps, y*+eps], 虚线标记目标 y*
    右侧: 命中率文字标注（各自落入 |LogP-1.0|<0.3 的占比）

数据源:
    generate/MOSES.txt   -> 无条件生成 (11887 行)
    cond/MOSESto1.txt    -> 条件生成, 目标 LogP=1 (5120 行)

用法:
    python redraw_hit_rate_conditional_vs_unconditional.py
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
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

# 脚本位于 todoimage/natureimg/scripts/
NATUREIMG_DIR = Path(__file__).resolve().parents[1]   # .../todoimage/natureimg
TODOIMAGE_DIR = NATUREIMG_DIR.parent                    # .../todoimage
OUTPUT_DIR = NATUREIMG_DIR / "images"

EXPORT_FORMATS = ["pdf", "svg", "eps", "tiff", "jpg"]
FORMAT_SUBDIRS = {fmt: fmt for fmt in EXPORT_FORMATS}
DPI = 600

TEXT_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"

UNCOND_COLOR = "#666666"       # 无条件生成: 灰色
UNCOND_FILL = "#E6E6E6"
COND_COLOR = "#08519C"         # 条件生成: 深蓝
COND_FILL = "#D6E6F2"          # 条件生成: 浅蓝填充(不透明)

TARGET = 1.0        # 目标 LogP y*
EPSILON = 0.3       # 命中阈值 sigma
SAMPLE_SIZE = 1000  # 每类抽取的有效分子数
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


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def logp_values(path: Path, sample_size: int, seed_offset: int = 0) -> np.ndarray:
    """固定种子洗牌后，收集 sample_size 个有效分子的 LogP。"""
    smiles = read_smiles(path)
    candidates = list(smiles)
    if len(candidates) > 1:
        rng = random.Random(SEED + seed_offset)
        rng.shuffle(candidates)

    values: List[float] = []
    for smi in candidates:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            values.append(float(Descriptors.MolLogP(mol)))
        except Exception:
            continue
        if len(values) >= sample_size:
            break
    return np.asarray(values, dtype=float)


def padded_range(arrays: Sequence[np.ndarray], pad_fraction: float = 0.08) -> Tuple[float, float]:
    merged = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if len(arr) > 0])
    low, high = float(np.min(merged)), float(np.max(merged))
    if np.isclose(low, high):
        return low - 1.0, high + 1.0
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def plot_kde_line(ax, values: np.ndarray, label: str, color: str, x_range: np.ndarray,
                  linestyle: str = "-", linewidth: float = 1.3, fill: bool = False,
                  fill_color: str | None = None) -> None:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return
    try:
        kde = gaussian_kde(values)
        y = kde(x_range)
    except Exception:
        return
    ax.plot(x_range, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label, zorder=3)
    if fill:
        ax.fill_between(x_range, y, 0, color=fill_color or color, alpha=1.0, linewidth=0, zorder=1)


def main() -> None:
    set_publication_style()

    uncond_path = TODOIMAGE_DIR / "generate" / "MOSES.txt"
    cond_path = TODOIMAGE_DIR / "cond" / "MOSESto1.txt"

    uncond = logp_values(uncond_path, SAMPLE_SIZE, seed_offset=0)
    cond = logp_values(cond_path, SAMPLE_SIZE, seed_offset=1)

    # 命中率: |LogP - TARGET| < EPSILON
    uncond_hit = float(np.mean(np.abs(uncond - TARGET) < EPSILON))
    cond_hit = float(np.mean(np.abs(cond - TARGET) < EPSILON))

    print(f"无条件生成: 有效分子 {len(uncond)} 个, 命中率 = {uncond_hit*100:.1f}%")
    print(f"条件生成:   有效分子 {len(cond)} 个, 命中率 = {cond_hit*100:.1f}%")

    fig, ax = plt.subplots(figsize=(6.8, 3.2))

    x_min, x_max = padded_range([uncond, cond])
    x_max = max(x_max, 8.0)
    x_range = np.linspace(x_min, x_max, 500)

    # 命中区间阴影 + 目标虚线
    ax.axvspan(TARGET - EPSILON, TARGET + EPSILON, color="#E8E8E8", alpha=1.0, linewidth=0, zorder=0)
    ax.axvline(TARGET, color=TEXT_COLOR, linestyle=":", linewidth=0.8, zorder=2)

    # 两条密度曲线
    plot_kde_line(ax, uncond, "Unconditional generation", UNCOND_COLOR, x_range,
                  linestyle="--", linewidth=1.1, fill=False)
    plot_kde_line(ax, cond, "Conditional (tree search)", COND_COLOR, x_range,
                  linestyle="-", linewidth=1.5, fill=True, fill_color=COND_FILL)

    ax.set_xlabel("LogP")
    ax.set_ylabel("Probability density")
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0)
    ax.tick_params(axis="both", pad=1.5)
    ax.set_ylim(bottom=0)

    # 右侧命中率文字标注
    hit_text = (
        f"Hit rate  (|LogP $-$ {TARGET:g}| < {EPSILON:g})\n"
        f"Unconditional:  {uncond_hit*100:.1f}%\n"
        f"Conditional:       {cond_hit*100:.1f}%"
    )
    fig.text(0.72, 0.50, hit_text, va="center", ha="left", fontsize=7.5,
             color=TEXT_COLOR, linespacing=1.6)

    ax.legend(loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.2, labelspacing=0.4)

    # 调整右侧留白给命中率文字
    fig.subplots_adjust(left=0.12, right=0.66, top=0.92, bottom=0.16)

    stem = "Fig_hit_rate_conditional_vs_unconditional"
    save_figure(fig, OUTPUT_DIR, stem, EXPORT_FORMATS)

    format_dirs = ", ".join(str(OUTPUT_DIR / FORMAT_SUBDIRS[fmt]) for fmt in EXPORT_FORMATS)
    print(f"\n[natureimg] 图片已按格式导出到: {format_dirs}")
    print(f"[natureimg] PNG 预览: {OUTPUT_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()
