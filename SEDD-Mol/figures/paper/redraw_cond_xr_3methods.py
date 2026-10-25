"""Redraw the cond_xr three-method LogP comparison in Nature-submission style.

This script produces one publication-ready figure from the SMILES files under
``todoimage/cond_xr``:

    new_{iter}.txt  -> SEDD-weighted sampling      (方法 A)
    low_{iter}.txt  -> Uniform-prior sampling      (方法 B)
    log_{iter}.txt  -> Prefix-conditioned sampling (前缀采样 N/NO/NN/NCO)

Each iteration node (0 / 10 / 100 / 200 / 500) is drawn as a single panel in
which the LogP kernel-density estimates of all three methods are overlaid on
the same axes, together with an optional grey dashed training-set reference
(default: ``todoimage/train/trainMOSES.txt``) and a dotted vertical line
marking the target LogP.

Layout, fonts, colours and multi-format export (pdf / svg / eps / tiff / jpg)
follow ``redraw_nature_figures.py``; results are written to
``natureimg/images/<fmt>/`` and captions / source-count summary rows are
appended to the shared ``figure_captions.txt`` and ``source_data_summary.csv``
files.

Usage (run from the scripts directory):

    python redraw_cond_xr_3methods.py
    python redraw_cond_xr_3methods.py --target-logp 1.0 --sample-size 5000
    python redraw_cond_xr_3methods.py --no-train
    python redraw_cond_xr_3methods.py --figure-no 11 --summary-mode append
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependency: "
        f"{exc.name}\n\n"
        "Install the required packages, for example:\n"
        "python -m pip install numpy matplotlib scipy rdkit\n"
    ) from exc

# Reuse the Nature-style helpers from the sibling script so that this figure
# shares exactly the same publication style, palette and export pipeline.
import redraw_nature_figures as base

PROJECT_ROOT = base.PROJECT_ROOT
DEFAULT_OUTPUT_DIR = base.DEFAULT_OUTPUT_DIR
COND_XR_DIR = PROJECT_ROOT / "cond_xr"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "train" / "trainMOSES.txt"

SEED = base.SEED
DPI = base.DPI
EXPORT_FORMATS = base.EXPORT_FORMATS
PANEL_LABELS = base.PANEL_LABELS

TRAIN_COLOR = "#A8A8A8"  # lighter grey for this figure (was #666666 in base)
TRAIN_FILL = base.TRAIN_FILL
GRID_COLOR = base.GRID_COLOR
TARGET_LINE_COLOR = base.TARGET_LINE_COLOR
TEXT_COLOR = base.TEXT_COLOR

# --- Palette ----------------------------------------------------------------
# Colour-blind-safe Okabe-Ito palette; the three methods stay distinct while
# remaining consistent with the publication style of the Fig1-Fig8 pipeline.
METHOD_COLORS = {
    "new": "#0072B2",   # deep blue:    SEDD-weighted
    "low": "#D55E00",   # vermillion:   Uniform-prior
    "log": "#009E73",   # bluish green: Prefix-conditioned
}
METHOD_LABELS = {
    "new": "SEDD-weighted",
    "low": "Uniform-weighted",
    "log": "Full-vocabulary UCT",
}
METHOD_ORDER = ["new", "low", "log"]
FIGURE_STEM = "Fig11_SEDD-weighted_vs_Uniform-weighted_vs_Full-vocabulary_UCT"


@dataclass
class _PanelStats:
    """KDE data for a single iteration panel (training + the three methods)."""

    train: base.ValuesWithStats
    methods: Dict[str, base.ValuesWithStats]
    iter_no: int


# --- Data loading -----------------------------------------------------------
def scan_cond_xr_files(data_dir: Path) -> Dict[str, Dict[int, Path]]:
    """Return {method: {iter: path}} for every ``{method}_{iter}.txt`` file."""
    files_by_method: Dict[str, Dict[int, Path]] = {m: {} for m in METHOD_ORDER}
    pattern = re.compile(r"(new|low|log)_(\d+)\.txt$")
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if not m:
            continue
        ftype, iters = m.group(1), int(m.group(2))
        files_by_method[ftype][iters] = data_dir / fname
    return files_by_method


def load_panel_stats(
    files_by_method: Dict[str, Dict[int, Path]],
    iter_no: int,
    train_path: Path | None,
    sample_size: int,
    figure_no: int,
) -> _PanelStats:
    train = (
        base.logp_values_from_file(train_path, sample_size, seed_offset=figure_no * 100)
        if train_path is not None
        else base.ValuesWithStats(
            np.empty((0,), dtype=float), raw_n=0, sampled_n=0, valid_n=0
        )
    )
    methods: Dict[str, base.ValuesWithStats] = {}
    for i, method in enumerate(METHOD_ORDER):
        path = files_by_method[method].get(iter_no)
        if path is None:
            methods[method] = base.ValuesWithStats(
                np.empty((0,), dtype=float), raw_n=0, sampled_n=0, valid_n=0
            )
            continue
        methods[method] = base.logp_values_from_file(
            path, sample_size, seed_offset=figure_no * 100 + i + 1
        )
    return _PanelStats(train=train, methods=methods, iter_no=iter_no)


# --- Figure ----------------------------------------------------------------
def plot_cond_xr_3methods(
    report: base.Report,
    output_dir: Path,
    figure_no: int,
    sample_size: int,
    target_logp: float,
    train_path: Path | None,
    formats: Sequence[str],
) -> None:
    """Fig11: one panel per iteration node, three methods overlaid per panel."""
    base.set_publication_style()

    files_by_method = scan_cond_xr_files(COND_XR_DIR)
    found = sum(len(v) for v in files_by_method.values())
    if found == 0:
        raise SystemExit(
            f"No new_*.txt / low_*.txt / log_*.txt files found in {COND_XR_DIR}"
        )

    all_iters = sorted(set().union(*(set(v) for v in files_by_method.values())))
    n_panels = len(all_iters)

    # Load per-panel data.
    panels: List[_PanelStats] = [
        load_panel_stats(files_by_method, it, train_path, sample_size, figure_no)
        for it in all_iters
    ]

    # Shared x-range over every curve.
    all_arrays = [p.train.values for p in panels]
    for p in panels:
        all_arrays.extend(v.values for v in p.methods.values())
    x_min, x_max = base.padded_range(all_arrays)
    x_range = np.linspace(x_min, x_max, 500)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(1.9 * n_panels, 2.3), sharex=True, sharey=True
    )
    axes_flat = [axes] if n_panels == 1 else list(np.ravel(axes))

    y_peaks: List[float] = []
    for i, (ax, panel) in enumerate(zip(axes_flat, panels)):
        # Training-set reference (grey dashed, light fill).
        if panel.train.valid_n > 0:
            y_peaks.append(
                base.plot_kde_line(
                    ax,
                    panel.train.values,
                    "Training set",
                    TRAIN_COLOR,
                    x_range,
                    linestyle="--",
                    linewidth=0.8,
                    fill=True,
                    fill_color=TRAIN_FILL,
                )
            )
        # The three generation methods.
        for method in METHOD_ORDER:
            stats = panel.methods[method]
            if stats.valid_n == 0:
                continue
            y_peaks.append(
                base.plot_kde_line(
                    ax,
                    stats.values,
                    METHOD_LABELS[method],
                    METHOD_COLORS[method],
                    x_range,
                    linestyle="-",
                    linewidth=0.9,
                    fill=False,
                )
            )
        # Target LogP marker.
        ax.axvline(target_logp, color=TARGET_LINE_COLOR, linestyle=":", linewidth=0.7)
        ax.set_title(f"Step {panel.iter_no}")
        base.style_density_axis(ax, "LogP", ylabel=(i == 0))
        base.add_panel_label(ax, PANEL_LABELS[i], x=-0.18, y=1.10)

    # Shared y-limit across panels.
    if y_peaks:
        shared_ymax = max(y_peaks) * 1.18
        for ax in axes_flat:
            ax.set_ylim(0, shared_ymax)

    # Figure-level legend.
    handles = [
        Line2D([0], [0], color=TRAIN_COLOR, linestyle="--", linewidth=0.8, label="Training set"),
        Line2D([0], [0], color=METHOD_COLORS["new"], linestyle="-", linewidth=0.9, label=METHOD_LABELS["new"]),
        Line2D([0], [0], color=METHOD_COLORS["low"], linestyle="-", linewidth=0.9, label=METHOD_LABELS["low"]),
        Line2D([0], [0], color=METHOD_COLORS["log"], linestyle="-", linewidth=0.9, label=METHOD_LABELS["log"]),
        Line2D([0], [0], color=TARGET_LINE_COLOR, linestyle=":", linewidth=0.7, label=f"Target LogP = {target_logp:g}"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.10), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    base.save_figure(fig, output_dir, FIGURE_STEM, formats=formats)

    # --- provenance records -------------------------------------------------
    figure_label = f"Fig. {figure_no}"
    for i, panel in enumerate(panels):
        if panel.train.valid_n > 0:
            report.add_row(
                base.SummaryRow(
                    figure=figure_label,
                    panel=PANEL_LABELS[i],
                    dataset="MOSES",
                    group="Training set",
                    source_file=str(train_path.relative_to(PROJECT_ROOT)),
                    raw_n=panel.train.raw_n,
                    sampled_n=panel.train.sampled_n,
                    valid_n=panel.train.valid_n,
                    note="Shared grey dashed LogP KDE reference.",
                )
            )
        for method in METHOD_ORDER:
            stats = panel.methods[method]
            if stats.valid_n == 0:
                continue
            report.add_row(
                base.SummaryRow(
                    figure=figure_label,
                    panel=PANEL_LABELS[i],
                    dataset="MOSES",
                    group=METHOD_LABELS[method],
                    source_file=str(files_by_method[method][panel.iter_no].relative_to(PROJECT_ROOT)),
                    raw_n=stats.raw_n,
                    sampled_n=stats.sampled_n,
                    valid_n=stats.valid_n,
                    note=f"LogP KDE at iteration step {panel.iter_no}; dotted line marks target LogP {target_logp:g}.",
                )
            )

    methods_text = ", ".join(METHOD_LABELS[m] for m in METHOD_ORDER)
    steps_text = ", ".join(f"{p.iter_no}" for p in panels)
    ref_text = (
        "The grey dashed curve denotes the training-set reference, and "
        if train_path is not None
        else ""
    )
    report.add_caption(
        figure_label,
        f"Conditional LogP distributions of molecules generated at successive sampling steps by three methods "
        f"({methods_text}). Panels correspond to iteration steps {steps_text}; "
        f"{ref_text}the dotted vertical line marks the target LogP {target_logp:g}. "
        "Kernel density estimates were computed from valid RDKit-parsed molecules; sample counts are provided in source_data_summary.csv.",
    )


# --- Summary IO -------------------------------------------------------------
_SUMMARY_FIELDS = [
    "figure",
    "panel",
    "dataset",
    "group",
    "source_file",
    "raw_n",
    "sampled_n",
    "valid_n",
    "note",
]


def _write_summary(report: base.Report, output_dir: Path, mode: str) -> None:
    """Append the new figure's records to figure_captions.txt / source_data_summary.csv.

    In ``append`` mode existing rows for the same figure are removed first so
    that repeated runs stay idempotent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    caption_path = output_dir / "figure_captions.txt"
    csv_path = output_dir / "source_data_summary.csv"
    new_figures = {fig for fig, _ in report.captions}

    fig_key_pattern = re.compile(r"^(Fig\.\s*\d+)\.\s*(.*)$", re.DOTALL)
    existing_captions: List[Tuple[str, str]] = []
    if mode == "append" and caption_path.exists():
        with caption_path.open("r", encoding="utf-8") as f:
            content = f.read()
        for block in filter(None, (b.strip() for b in content.split("\n\n"))):
            match = fig_key_pattern.match(block)
            if not match:
                continue
            fig_key = match.group(1).strip()
            if fig_key in new_figures:
                continue
            existing_captions.append((fig_key, match.group(2).strip()))

    with caption_path.open("w", encoding="utf-8") as f:
        for fig_key, caption_body in existing_captions + report.captions:
            f.write(f"{fig_key}. {caption_body}\n\n")

    existing_rows: List[Dict[str, str]] = []
    if mode == "append" and csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("figure") in new_figures:
                    continue
                existing_rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in _SUMMARY_FIELDS})
        for row in report.rows:
            writer.writerow({k: getattr(row, k) for k in _SUMMARY_FIELDS})


# --- CLI --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw the cond_xr three-method (new/low/log) LogP comparison "
            "in the unified Nature-submission style."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported figure files (shared with the Fig1-Fig10 pipeline).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Target number of valid molecules sampled per group; if fewer are available, all valid molecules are used.",
    )
    parser.add_argument(
        "--target-logp",
        type=float,
        default=1.0,
        help="Target LogP value drawn as a dotted vertical line.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Optional training-set SMILES file drawn as grey dashed reference (e.g. todoimage/train/trainMOSES.txt). By default no training-set curve is drawn.",
    )
    parser.add_argument(
        "--figure-no",
        type=int,
        default=11,
        help="Figure number used in the output stem (default 11).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "svg", "eps", "tiff", "jpg"],
        default=EXPORT_FORMATS,
        help="Export formats. Same choices as the Fig1-Fig10 pipeline.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=["append", "overwrite"],
        default="append",
        help=(
            "How to update figure_captions.txt / source_data_summary.csv. "
            "'append' keeps existing Fig1-Fig10 records; 'overwrite' rewrites both files with only this figure."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path: Path | None = args.train
    if train_path is not None and not train_path.exists():
        print(f"[warning] Training set not found, skipping reference: {train_path}")
        train_path = None

    report = base.Report()
    plot_cond_xr_3methods(
        report=report,
        output_dir=output_dir,
        figure_no=args.figure_no,
        sample_size=args.sample_size,
        target_logp=args.target_logp,
        train_path=train_path,
        formats=args.formats,
    )
    _write_summary(report, output_dir, mode=args.summary_mode)
    print(
        f"Saved Fig{args.figure_no} (cond_xr 3-method LogP comparison) to {output_dir} "
        f"in formats: {', '.join(args.formats)}."
    )
    print(
        f"Updated {output_dir / 'figure_captions.txt'} and "
        f"{output_dir / 'source_data_summary.csv'} (mode={args.summary_mode})."
    )


if __name__ == "__main__":
    main()
