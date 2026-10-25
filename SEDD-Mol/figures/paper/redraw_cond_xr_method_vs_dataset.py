"""Redraw the SEDD-weighted method vs. the training dataset in Nature style.

Same plotting pipeline and style as ``redraw_cond_xr_3methods.py`` (Fig11):
one panel per iteration node, but with only *two* curves per panel:

    new_{iter}.txt  -> SEDD-weighted sampling  (our method, solid blue)
    trainMOSES.txt  -> training dataset        (grey dashed reference)

A dotted vertical line marks the target LogP. Fonts, colours, layout and
multi-format export (pdf / svg / eps / tiff / jpg) follow
``redraw_nature_figures.py``; results are written to ``natureimg/images/<fmt>/``
and captions / source-count summary rows are appended to the shared
``figure_captions.txt`` and ``source_data_summary.csv`` files.

Usage (run from the scripts directory):

    python redraw_cond_xr_method_vs_dataset.py
    python redraw_cond_xr_method_vs_dataset.py --target-logp 1.0 --sample-size 5000
    python redraw_cond_xr_method_vs_dataset.py --figure-no 13
"""

from __future__ import annotations

import argparse
import csv
import os
import re
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

TRAIN_COLOR = base.TRAIN_COLOR
TRAIN_FILL = base.TRAIN_FILL
TARGET_LINE_COLOR = base.TARGET_LINE_COLOR

METHOD_COLOR = "#0072B2"   # deep blue: SEDD-weighted (our method)
METHOD_LABEL = "SEDD-weighted"
DATASET_LABEL = "Dataset (train)"
FIGURE_STEM = "Fig13_SEDD-weighted_vs_dataset"


# --- Data loading -----------------------------------------------------------
def scan_new_files(data_dir: Path) -> Dict[int, Path]:
    """Return {iter: path} for every ``new_{iter}.txt`` file."""
    files_by_iter: Dict[int, Path] = {}
    pattern = re.compile(r"new_(\d+)\.txt$")
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if not m:
            continue
        files_by_iter[int(m.group(1))] = data_dir / fname
    return files_by_iter


# --- Figure ----------------------------------------------------------------
def plot_method_vs_dataset(
    report: base.Report,
    output_dir: Path,
    figure_no: int,
    sample_size: int,
    target_logp: float,
    train_path: Path,
    formats: Sequence[str],
) -> None:
    """One panel per iteration node; two curves per panel: method vs. dataset."""
    base.set_publication_style()

    files_by_iter = scan_new_files(COND_XR_DIR)
    if not files_by_iter:
        raise SystemExit(f"No new_*.txt files found in {COND_XR_DIR}")

    all_iters = sorted(files_by_iter.keys())
    n_panels = len(all_iters)

    # Load data.
    train = base.logp_values_from_file(train_path, sample_size, seed_offset=figure_no * 100)
    steps = {
        it: base.logp_values_from_file(
            files_by_iter[it], sample_size, seed_offset=figure_no * 100 + i + 1
        )
        for i, it in enumerate(all_iters)
    }

    # Shared x-range over every curve.
    all_arrays = [train.values] + [s.values for s in steps.values()]
    x_min, x_max = base.padded_range(all_arrays)
    x_range = np.linspace(x_min, x_max, 500)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(1.9 * n_panels, 2.3), sharex=True, sharey=True
    )
    axes_flat = [axes] if n_panels == 1 else list(np.ravel(axes))

    y_peaks: List[float] = []
    for i, (ax, it) in enumerate(zip(axes_flat, all_iters)):
        # Dataset reference (grey dashed, light fill).
        y_peaks.append(
            base.plot_kde_line(
                ax,
                train.values,
                DATASET_LABEL,
                TRAIN_COLOR,
                x_range,
                linestyle="--",
                linewidth=0.8,
                fill=True,
                fill_color=TRAIN_FILL,
            )
        )
        # Our method (solid blue).
        stats = steps[it]
        if stats.valid_n > 0:
            y_peaks.append(
                base.plot_kde_line(
                    ax,
                    stats.values,
                    METHOD_LABEL,
                    METHOD_COLOR,
                    x_range,
                    linestyle="-",
                    linewidth=0.9,
                    fill=False,
                )
            )
        # Target LogP marker.
        ax.axvline(target_logp, color=TARGET_LINE_COLOR, linestyle=":", linewidth=0.7)
        ax.set_title(f"Step {it}")
        base.style_density_axis(ax, "LogP", ylabel=(i == 0))
        base.add_panel_label(ax, PANEL_LABELS[i], x=-0.18, y=1.10)

    # Shared y-limit across panels.
    if y_peaks:
        shared_ymax = max(y_peaks) * 1.18
        for ax in axes_flat:
            ax.set_ylim(0, shared_ymax)

    # Figure-level legend (two curves only).
    handles = [
        Line2D([0], [0], color=METHOD_COLOR, linestyle="-", linewidth=0.9, label=METHOD_LABEL),
        Line2D([0], [0], color=TRAIN_COLOR, linestyle="--", linewidth=0.8, label=DATASET_LABEL),
        Line2D([0], [0], color=TARGET_LINE_COLOR, linestyle=":", linewidth=0.7, label=f"Target LogP = {target_logp:g}"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    base.save_figure(fig, output_dir, FIGURE_STEM, formats=formats)

    # --- provenance records -------------------------------------------------
    figure_label = f"Fig. {figure_no}"
    for i, it in enumerate(all_iters):
        report.add_row(
            base.SummaryRow(
                figure=figure_label,
                panel=PANEL_LABELS[i],
                dataset="MOSES",
                group=DATASET_LABEL,
                source_file=str(train_path.relative_to(PROJECT_ROOT)),
                raw_n=train.raw_n,
                sampled_n=train.sampled_n,
                valid_n=train.valid_n,
                note="Shared grey dashed LogP KDE reference.",
            )
        )
        stats = steps[it]
        report.add_row(
            base.SummaryRow(
                figure=figure_label,
                panel=PANEL_LABELS[i],
                dataset="MOSES",
                group=METHOD_LABEL,
                source_file=str(files_by_iter[it].relative_to(PROJECT_ROOT)),
                raw_n=stats.raw_n,
                sampled_n=stats.sampled_n,
                valid_n=stats.valid_n,
                note=f"LogP KDE at iteration step {it}; dotted line marks target LogP {target_logp:g}.",
            )
        )

    steps_text = ", ".join(f"{it}" for it in all_iters)
    report.add_caption(
        figure_label,
        f"Conditional LogP distributions of molecules generated by SEDD-weighted sampling "
        f"({METHOD_LABEL}) compared with the training dataset. Panels correspond to iteration "
        f"steps {steps_text}; the grey dashed curve denotes the dataset reference, and the dotted "
        f"vertical line marks the target LogP {target_logp:g}. Kernel density estimates were "
        "computed from valid RDKit-parsed molecules; sample counts are provided in source_data_summary.csv.",
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
            "Redraw the SEDD-weighted method vs. training dataset LogP comparison "
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
        default=DEFAULT_TRAIN_PATH,
        help="Training dataset SMILES file (default: todoimage/train/trainMOSES.txt).",
    )
    parser.add_argument(
        "--figure-no",
        type=int,
        default=13,
        help="Figure number used in the output stem (default 13).",
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

    train_path: Path = args.train
    if not train_path.exists():
        raise SystemExit(f"Training dataset not found: {train_path}")

    report = base.Report()
    plot_method_vs_dataset(
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
        f"Saved Fig{args.figure_no} (SEDD-weighted vs dataset) to {output_dir} "
        f"in formats: {', '.join(args.formats)}."
    )
    print(
        f"Updated {output_dir / 'figure_captions.txt'} and "
        f"{output_dir / 'source_data_summary.csv'} (mode={args.summary_mode})."
    )


if __name__ == "__main__":
    main()
