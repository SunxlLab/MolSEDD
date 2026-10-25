"""
Nature 投稿风格的消融实验折线图（4 条线）。

基于 redraw_nature_figures.py 的出版级样式重构：
    - Arial / Helvetica 字体, 7pt 正文
    - 细轴线 (0.7pt), 去 top/right spine
    - 无框图例, 浅灰网格
    - 600 DPI, 多格式导出 (pdf / svg / eps / tiff / jpg)

图内容（前 500 次评估的 best-so-far reward R）:
    三条 UCT 基线（灰阶虚线）:
        1. Static-vocab UCT (Pure MCTS)
        2. Dynamic-vocab UCT, no validity (MOSES_5)
        3. Dynamic-vocab UCT (GGTMCTS)
    一条 PUCT 主方法（深蓝实线, 加粗）:
        4. Frequency PUCT + validity (ours)

数据源（项目根目录）:
    condition_log_MOSES/PureMCTS_to1.txt
    condition_log_MOSES/GGTMCTSto1.txt
    ablation_results/MOSES_1_our/run.log
    ablation_results/MOSES_2_减去频率/run.log
    ablation_results/MOSES_3_减去有效性检测/run.log
    ablation_results/MOSES_4_减去有效性检测和频率/run.log
    ablation_results/MOSES_5_UCT无有效性检测/run.log

用法:
    python redraw_ablation_7lines.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 备用中文字体（若图例含中文时 fallback）
for _fp in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]:
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception:
        pass

# 脚本位于 todoimage/natureimg/scripts/, 因此:
NATUREIMG_DIR = Path(__file__).resolve().parents[1]   # .../todoimage/natureimg
REPO_ROOT = Path(__file__).resolve().parents[3]        # .../Score-Entropy-Discrete-Diffusion
OUTPUT_DIR = NATUREIMG_DIR / "images"

EXPORT_FORMATS = ["pdf", "svg", "eps", "tiff", "jpg"]
FORMAT_SUBDIRS = {fmt: fmt for fmt in EXPORT_FORMATS}
DPI = 600

# ---- 出版级配色 ----
TEXT_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"

# 三条 UCT 基线（灰阶, 由弱到强）
UCT_COLORS = ["#C7C7C7", "#9A9A9A", "#5A5A5A"]
# 四条 PUCT（ColorBrewer Blues, 由浅到深, 主方法最深）
PUCT_COLORS = ["#9ECAE1", "#6BAED6", "#4292C6", "#08519C"]

MAX_DISC = 500

# (图例名, 日志路径, 颜色, 线型, 线宽)
SOURCES: List[Tuple[str, Path, str, str, float]] = [
    ("Ours",                            REPO_ROOT / "ablation_results" / "MOSES_1_our" / "run.log",             PUCT_COLORS[3], "-", 1.8),
    ("MCTS + dynamic vocab + validity", REPO_ROOT / "condition_log_MOSES" / "GGTMCTSto1.txt",                   UCT_COLORS[2], "--", 1.2),
    ("MCTS + dynamic vocab",            REPO_ROOT / "ablation_results" / "MOSES_5_UCT无有效性检测" / "run.log", UCT_COLORS[1], "--", 1.2),
    ("Pure MCTS",                       REPO_ROOT / "condition_log_MOSES" / "PureMCTS_to1.txt",                UCT_COLORS[0], "--", 1.2),
]

DISC_RE = re.compile(
    r"\[Disc & Sim\] Prefix: .*? \| Eff: .*? \| AvgLogP: (-?[0-9.]+) \| MAE: .*? \| R: ([0-9.]+)"
)
LOOP_RE = re.compile(r"Loop (\d+)/\d+ \| R: ([0-9.]+)")


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
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
    # 额外存一份 PNG 预览到 images 根目录，便于快速查看
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, **common)
    plt.close(fig)


def extract_from_disc(log_path: Path, max_disc: int = MAX_DISC):
    best_r = -1.0
    series = []
    disc_idx = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = DISC_RE.search(line)
            if not m:
                continue
            avg_logp = float(m.group(1))
            r = float(m.group(2))
            disc_idx += 1
            if disc_idx > max_disc:
                break
            if avg_logp > -90 and r > best_r:
                best_r = r
            series.append((disc_idx, best_r))
    return series


def extract_from_loop(log_path: Path, max_step: int = MAX_DISC):
    best_r = -1.0
    series = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOOP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            r = float(m.group(2))
            if step > max_step:
                break
            if r > best_r:
                best_r = r
            series.append((step, best_r))
    return series


def main() -> None:
    set_publication_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    curves = []
    for label, log_path, color, ls, lw in SOURCES:
        if not log_path.exists():
            print(f"[skip] {log_path} 不存在")
            continue
        if "PureMCTS" in log_path.name:
            series = extract_from_loop(log_path)
        else:
            series = extract_from_disc(log_path)
        if not series:
            print(f"[skip] {label} 无数据")
            continue
        steps = [s for s, _ in series]
        rs = [r for _, r in series]
        curves.append((label, steps, rs, color, ls, lw))
        print(f"{label}: 最终 R = {rs[-1]:.4f}")

    # 画曲线
    for label, steps, rs, color, ls, lw in curves:
        ax.plot(steps, rs, color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)

    # 末端数值标注: 数字贴线终点, 仅对过近的相邻数字做最小错开
    ordered = sorted(curves, key=lambda c: c[2][-1], reverse=True)
    MIN_GAP = 0.024
    last_y = None
    for (label, steps, rs, color, ls, lw) in ordered:
        y_end = rs[-1]
        place_y = y_end
        if last_y is not None and (last_y - place_y) < MIN_GAP:
            place_y = last_y - MIN_GAP
        ax.text(
            steps[-1] + 8, place_y, f"{y_end:.3f}",
            color=color, fontsize=6.5, va="center", ha="left", zorder=4,
        )
        last_y = place_y

    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Best-so-far reward R")
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0)
    ax.tick_params(axis="both", pad=1.5)
    ax.set_xlim(left=0, right=575)
    ax.set_ylim(bottom=0.2)

    ax.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2, labelspacing=0.4)

    stem = "Fig_ablation_best_so_far_reward_4lines"
    save_figure(fig, OUTPUT_DIR, stem, EXPORT_FORMATS)

    format_dirs = ", ".join(str(OUTPUT_DIR / FORMAT_SUBDIRS[fmt]) for fmt in EXPORT_FORMATS)
    print(f"\n[natureimg] 图片已按格式导出到: {format_dirs}")
    print(f"[natureimg] PNG 预览: {OUTPUT_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()
