"""Redraw the cond_generated LogP evolution in the style of plot_xr_comparison.py.

The figure shows *every* selected iteration node from
``cond_generated/{loop}-{prefix}/molecules.txt``, one panel per node, arranged
left-to-right by iteration count. Each panel overlays the generated LogP KDE
curve on the training-set reference and a dotted target line.

The plotting code follows ``todoimage/plot_xr_comparison.py`` /
``plot_xr_log_comparison.py`` as closely as possible: figure size, colours,
line widths, fill alpha, titles (``Step {loop}``), axis labels, target line,
grid and legend style are copied from those scripts.

Usage (run from the scripts directory):

    python redraw_cond_generated_logp_evolution.py --only-loops 0 10 50 120 280
    python redraw_cond_generated_logp_evolution.py --max-loop 1030
    python redraw_cond_generated_logp_evolution.py --output-dir /path/to/dir
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing plotting dependency: {exc.name}") from exc

import redraw_nature_figures as base

PROJECT_ROOT = base.PROJECT_ROOT
REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo> root
COND_GENERATED_DIR = REPO_ROOT / "cond_generated"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "train" / "trainMOSES.txt"


def scan_cond_generated(
    data_dir: Path, max_loop: Optional[int], only_loops: Optional[List[int]] = None
) -> List[tuple[int, str, Path]]:
    """Return [(loop, prefix, molecules_path), ...] sorted by loop, from {loop}-{prefix}/ dirs."""
    items: List[tuple[int, str, Path]] = []
    pattern = re.compile(r"^(\d+)-(.+)$")
    only_set = set(only_loops) if only_loops else None
    if not data_dir.exists():
        raise SystemExit(f"cond_generated directory not found: {data_dir}")
    for fname in sorted(data_dir.iterdir()):
        if not fname.is_dir():
            continue
        m = pattern.match(fname.name)
        if not m:
            continue
        loop = int(m.group(1))
        if max_loop is not None and loop > max_loop:
            print(f"  [skip] {fname.name}: loop {loop} > max_loop={max_loop}")
            continue
        if only_set is not None and loop not in only_set:
            print(f"  [skip] {fname.name}: loop {loop} not in only_loops")
            continue
        mol_path = fname / "molecules.txt"
        if not mol_path.exists():
            print(f"  [skip] {fname.name}: no molecules.txt")
            continue
        items.append((loop, m.group(2), mol_path))
    items.sort(key=lambda x: x[0])
    return items


def plot_kde(ax, values, label, color, linestyle='-', fill=True):
    """完全照抄 plot_xr_comparison.py 的 plot_kde 画法。"""
    if len(values) < 2:
        return
    try:
        kde = gaussian_kde(values)
        pad = (max(values) - min(values)) * 0.15 or 1.0
        x_range = np.linspace(min(values) - pad, max(values) + pad, 500)
        y_kde = kde(x_range)
        ax.plot(x_range, y_kde, linewidth=2, label=label, color=color, linestyle=linestyle)
        if fill:
            ax.fill_between(x_range, y_kde, alpha=0.1, color=color)
    except Exception as e:
        print(f"    KDE 绘制失败: {e}")


def plot_cond_generated_logp_evolution(
    output_dir: Path,
    data_dir: Path,
    train_path: Path,
    max_loop: Optional[int],
    only_loops: Optional[List[int]],
    sample_size: int,
    target_logp: float,
    figure_no: int,
    formats: Sequence[str],
) -> None:
    items = scan_cond_generated(data_dir, max_loop, only_loops)
    if len(items) == 0:
        raise SystemExit("No {loop}-{prefix}/molecules.txt found.")
    print(f"Found {len(items)} iteration nodes.")

    train = base.logp_values_from_file(train_path, sample_size, seed_offset=figure_no * 100)
    steps = {
        loop: base.logp_values_from_file(path, sample_size, seed_offset=figure_no * 100 + i + 1)
        for i, (loop, _, path) in enumerate(items)
    }
    prefixes = {loop: prefix for loop, _, prefix in items}

    n_panels = len(items)

    # ---- 绘图：完全参考 plot_xr_comparison.py / plot_xr_log_comparison.py ----
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # ---- 单排叠加：所有迭代节点横排，每个 panel 一张分布图 ----
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    color_gen = '#2874A6'     # 蓝色：与参考文件 SEDD-weighted 同色
    train_color = '#666666'

    for col, (loop, _prefix, _path) in enumerate(items):
        ax = axes[col]

        # 训练集参考线（灰色虚线）
        if len(train.values) > 0:
            plot_kde(ax, train.values, "Train", train_color, linestyle='--', fill=True)

        # 当前迭代 LogP 分布（蓝色实线，与参考文件同粗细/同填充）
        item = steps[loop]
        if len(item.values) > 0:
            plot_kde(ax, item.values, f"Step {loop}", color_gen, fill=True)

        ax.set_title(f"Step {loop}", fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel("LogP", fontsize=11)

        # 目标竖虚线
        ax.axvline(x=target_logp, color='black', linestyle='--', linewidth=1.3, alpha=0.7,
                   label=f'Target={target_logp}')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', frameon=True, fontsize=9)

    axes[0].set_ylabel("Probability Density", fontsize=11)

    plt.tight_layout()

    stem = f"Fig{figure_no}_cond_generated_logp_evolution"
    # 输出 png + pdf（600 dpi，与 plot_xr_log_comparison.py 一致）
    (output_dir / "pdf").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "pdf" / f"{stem}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches='tight')
    # 保留 natureimg 的多格式导出
    base.save_figure(fig, output_dir, stem, formats=formats)

    # Provenance summary (append-friendly: only rewrite records for this figure).
    caption = (
        f"Evolution of conditional LogP distributions of generated molecules toward the target value {target_logp:g}. "
        "Panels show the generated distribution at successive MCTS search steps overlaid on the training-set reference "
        f"(MOSES). The dotted vertical line marks the target LogP. Kernel density estimates were computed from valid "
        "RDKit-parsed molecules; sample counts are provided in source_data_summary.csv."
    )
    write_provenance(output_dir, figure_no, train, steps, prefixes, train_path, items, caption)


def write_provenance(
    output_dir: Path,
    figure_no: int,
    train,
    steps: Dict[int, base.ValuesWithStats],
    prefixes: Dict[int, str],
    train_path: Path,
    items: List[tuple[int, str, Path]],
    caption: str,
) -> None:
    figure_label = f"Fig. {figure_no}"
    # Remove existing records for this figure, then append new ones.
    caption_path = output_dir / "figure_captions.txt"
    csv_path = output_dir / "source_data_summary.csv"

    if caption_path.exists():
        blocks = [b.strip() for b in caption_path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
        keep = [b for b in blocks if not b.startswith(f"{figure_label}.")]
        caption_path.write_text("\n\n".join(keep) + ("\n\n" if keep else ""), encoding="utf-8")
    with caption_path.open("a", encoding="utf-8") as f:
        f.write(f"{figure_label}. {caption}\n")

    header = ["figure", "panel", "dataset", "group", "source_file", "raw_n", "sampled_n", "valid_n", "note"]
    import csv as _csv
    existing = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                if row.get("figure") != figure_label:
                    existing.append(row)
    rows = existing
    rows.append({
        "figure": figure_label, "panel": "all", "dataset": "MOSES", "group": "Training set",
        "source_file": str(train_path.relative_to(PROJECT_ROOT)),
        "raw_n": train.raw_n, "sampled_n": train.sampled_n, "valid_n": train.valid_n,
        "note": "Shared grey dashed reference across all panels.",
    })
    for i, (loop, _, path) in enumerate(items):
        step_name = f"Step {loop}"
        item = steps[loop]
        rows.append({
            "figure": figure_label, "panel": base.PANEL_LABELS[i], "dataset": "MOSES",
            "group": f"{step_name} ({prefixes[loop]})",
            "source_file": str(path.relative_to(REPO_ROOT)),
            "raw_n": item.raw_n, "sampled_n": item.sampled_n, "valid_n": item.valid_n,
            "note": f"Generated molecules at {step_name}; dotted line marks target LogP.",
        })
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw cond_generated LogP evolution in Fig3 style."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "natureimg" / "images",
        help="Directory for exported figure files (default: natureimg/images).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=COND_GENERATED_DIR,
        help="Directory containing {loop}-{prefix}/molecules.txt (default: <repo>/cond_generated).",
    )
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_TRAIN_PATH,
        help="Training-set SMILES file (default: todoimage/train/trainMOSES.txt).",
    )
    parser.add_argument(
        "--max-loop", type=int, default=None,
        help="Only include nodes with loop <= this value (e.g. 1030).",
    )
    parser.add_argument(
        "--only-loops", type=int, nargs="+", default=None,
        help="Only include these exact loop values, e.g. --only-loops 0 10 50 120 460.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=3000,
        help="Target valid molecules per node.",
    )
    parser.add_argument(
        "--target-logp", type=float, default=1.0,
        help="Target LogP value drawn as dotted vertical line.",
    )
    parser.add_argument(
        "--figure-no", type=int, default=12,
        help="Figure number used in output stem and provenance records.",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["pdf", "svg", "eps", "tiff", "jpg"],
        default=base.EXPORT_FORMATS, help="Export formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train is not None and not args.train.exists():
        print(f"[warning] Training set not found: {args.train}")
        args.train = None
    plot_cond_generated_logp_evolution(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        train_path=args.train,
        max_loop=args.max_loop,
        only_loops=args.only_loops,
        sample_size=args.sample_size,
        target_logp=args.target_logp,
        figure_no=args.figure_no,
        formats=args.formats,
    )
    print(f"Saved Fig{args.figure_no} to {args.output_dir} in {', '.join(args.formats)}.")


if __name__ == "__main__":
    main()
