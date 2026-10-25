"""Redraw LZPneed SA/RA figures in the unified Nature-submission style.

This script produces two publication-ready figures from the CSVs under
``todoimage/LZPneed`` and the training SMILES under ``todoimage/train``:

* ``Fig9_SA_RA_scatter_2x2``: 2x2 scatter of SA score vs RA score for one
  unconditional and three conditional (LogP = 1/2/3) MOSES generations,
  coloured by whether each SMILES is identical to a training-set molecule.
* ``Fig10_SA_RA_KDE_2panel``: 2-panel KDE curves comparing the training
  set with unconditional / conditional generations on SA score (left) and
  RA score (right).

Layout, fonts, colours and multi-format export (pdf / svg / eps / tiff /
jpg) follow ``redraw_nature_figures.py``; results are written to
``natureimg/images/<fmt>/`` and captions / source-count summary rows are
appended to the shared ``figure_captions.txt`` and
``source_data_summary.csv`` files.
"""

from __future__ import annotations

import argparse
import csv
import random
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
    from scipy.stats import gaussian_kde
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependency: "
        f"{exc.name}\n\n"
        "Install the required packages, for example:\n"
        "python -m pip install numpy matplotlib scipy\n"
    ) from exc

# Reuse the Nature-style helpers from the sibling script so that Fig9/Fig10
# share exactly the same publication style, palette and export pipeline.
import redraw_nature_figures as base

PROJECT_ROOT = base.PROJECT_ROOT
DEFAULT_OUTPUT_DIR = base.DEFAULT_OUTPUT_DIR
LZPNEED_DIR = PROJECT_ROOT / "LZPneed"
TRAIN_SMILES_PATH = PROJECT_ROOT / "train" / "trainMOSES.txt"

SEED = base.SEED
DPI = base.DPI
EXPORT_FORMATS = base.EXPORT_FORMATS
PANEL_LABELS = base.PANEL_LABELS

TRAIN_COLOR = base.TRAIN_COLOR
TRAIN_FILL = base.TRAIN_FILL
GRID_COLOR = base.GRID_COLOR

# --- Palette ----------------------------------------------------------------
# Fig9: two-class scatter (novel vs. identical-to-training).
NOVEL_COLOR = "#1F5AA6"          # deep blue: de novo novel molecules
IDENTICAL_COLOR = "#8E2C2C"      # deep red:  identical to training set

# Fig10: 5 KDE curves. Training grey dashed; the four generated curves are a
# perceptually ordered blue-to-purple gradient consistent with Fig. 3.
GENERATION_COLORS = {
    "Unconditional": "#9ECAE1",
    "LogP = 1": "#4292C6",
    "LogP = 2": "#08519C",
    "LogP = 3": "#6A51A3",
}


@dataclass
class SAScatterStats:
    """Sampled SA/RA points partitioned by training-set membership."""

    in_train_sa: np.ndarray
    in_train_ra: np.ndarray
    out_train_sa: np.ndarray
    out_train_ra: np.ndarray
    raw_n: int
    sampled_n: int
    valid_n: int

    @property
    def in_train_n(self) -> int:
        return int(self.in_train_sa.size)

    @property
    def out_train_n(self) -> int:
        return int(self.out_train_sa.size)


@dataclass
class MetricValues:
    """Sampled 1D metric values with provenance counts."""

    values: np.ndarray
    raw_n: int
    sampled_n: int
    valid_n: int


# --- Data loading -----------------------------------------------------------
def _iter_valid_rows(csv_path: Path):
    """Yield (smiles, sa, ra) tuples for rows with ``valid == 1``."""
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("valid") or "").strip() != "1":
                continue
            smi = (row.get("smiles") or "").strip()
            sa_raw = (row.get("sa_score") or "").strip()
            ra_raw = (row.get("ra_score") or "").strip()
            if not smi or not sa_raw or not ra_raw:
                continue
            try:
                sa = float(sa_raw)
                ra = float(ra_raw)
            except ValueError:
                continue
            yield smi, sa, ra


def load_train_smiles(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"Training SMILES file not found: {path}")
    smiles = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles.add(s)
    return smiles


def _count_csv_rows(csv_path: Path) -> int:
    """Return the number of data rows in a CSV (header excluded)."""
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in f)
    return max(total - 1, 0)


def load_sa_ra_scatter(
    csv_path: Path,
    train_smiles: set,
    sample_size: int,
    seed_offset: int,
) -> SAScatterStats:
    """Sample up to ``sample_size`` valid SA/RA rows and split by train membership."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    records: List[Tuple[str, float, float]] = list(_iter_valid_rows(csv_path))
    raw_n = _count_csv_rows(csv_path)
    valid_n = len(records)

    if valid_n == 0:
        empty = np.empty((0,), dtype=float)
        return SAScatterStats(empty, empty, empty, empty, raw_n, 0, 0)

    use_n = min(sample_size, valid_n)
    rng = random.Random(SEED + seed_offset)
    sampled = rng.sample(records, use_n)

    in_sa, in_ra, out_sa, out_ra = [], [], [], []
    for smi, sa, ra in sampled:
        if smi in train_smiles:
            in_sa.append(sa)
            in_ra.append(ra)
        else:
            out_sa.append(sa)
            out_ra.append(ra)

    return SAScatterStats(
        in_train_sa=np.asarray(in_sa, dtype=float),
        in_train_ra=np.asarray(in_ra, dtype=float),
        out_train_sa=np.asarray(out_sa, dtype=float),
        out_train_ra=np.asarray(out_ra, dtype=float),
        raw_n=raw_n,
        sampled_n=use_n,
        valid_n=valid_n,
    )


def load_metric_values(
    csv_path: Path,
    metric_key: str,
    sample_size: int,
    seed_offset: int,
) -> MetricValues:
    """Sample up to ``sample_size`` valid values of ``metric_key`` from a CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw_n = _count_csv_rows(csv_path)
    values: List[float] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("valid") or "").strip() != "1":
                continue
            raw = (row.get(metric_key) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue

    valid_n = len(values)
    if valid_n == 0:
        return MetricValues(np.empty((0,), dtype=float), raw_n, 0, 0)

    use_n = min(sample_size, valid_n)
    rng = random.Random(SEED + seed_offset)
    sampled = rng.sample(values, use_n) if valid_n > use_n else values
    return MetricValues(
        values=np.asarray(sampled, dtype=float),
        raw_n=raw_n,
        sampled_n=use_n,
        valid_n=valid_n,
    )


# --- Plot helpers -----------------------------------------------------------
def _bounded_kde(
    values: np.ndarray,
    x_min: float,
    x_max: float,
    n_points: int = 800,
    bandwidth_scale: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a smooth KDE curve on a fixed grid; robust to small samples."""
    vals = values[np.isfinite(values)]
    vals = vals[(vals >= x_min - 0.5) & (vals <= x_max + 0.5)]
    x = np.linspace(x_min, x_max, n_points)
    if vals.size < 2:
        return x, np.zeros(n_points)
    try:
        kde = gaussian_kde(
            vals,
            bw_method=lambda s: s.scotts_factor() * bandwidth_scale,
        )
        return x, kde(x)
    except Exception:
        return x, np.zeros(n_points)


def _draw_scatter_panel(
    ax,
    stats: SAScatterStats,
    title: str,
    panel_label: str | None,
) -> None:
    ax.scatter(
        stats.out_train_sa,
        stats.out_train_ra,
        s=6.5,
        c=NOVEL_COLOR,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        label=f"De novo novel (n={stats.out_train_n})",
    )
    ax.scatter(
        stats.in_train_sa,
        stats.in_train_ra,
        s=7.5,
        c=IDENTICAL_COLOR,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        label=f"Identical to training (n={stats.in_train_n})",
    )

    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("SA score (\u2193)")
    ax.set_ylabel("RA score (\u2191)")
    ax.set_title(title)
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0)
    ax.tick_params(axis="both", pad=1.5)
    ax.legend(
        loc="lower right",
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.3,
        markerscale=1.8,
    )
    if panel_label is not None:
        base.add_panel_label(ax, panel_label)


def _draw_kde_panel(
    ax,
    curves: Sequence[Tuple[str, MetricValues, str, str, float]],
    x_min: float,
    x_max: float,
    x_label: str,
    xticks: Sequence[float] | None,
    bandwidth_scale: float,
    panel_label: str | None,
) -> None:
    y_max = 0.0
    for label, metric, color, linestyle, linewidth in curves:
        if metric.values.size == 0:
            continue
        x, y = _bounded_kde(
            metric.values,
            x_min=x_min,
            x_max=x_max,
            n_points=800,
            bandwidth_scale=bandwidth_scale,
        )
        is_training = label == "Training set"
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=6 if is_training else 5,
        )
        ax.fill_between(
            x,
            0,
            y,
            color=TRAIN_FILL if is_training else color,
            alpha=0.10 if is_training else 0.06,
            linewidth=0,
            zorder=1,
        )
        y_max = max(y_max, float(np.nanmax(y)) if y.size else 0.0)

    ax.set_xlim(x_min, x_max)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    ax.set_ylim(0.0, y_max * 1.15 if y_max > 0 else None)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability density")
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0)
    ax.tick_params(axis="both", pad=1.5)
    ax.legend(
        loc="upper right",
        handlelength=1.6,
        handletextpad=0.45,
        borderaxespad=0.3,
    )
    if panel_label is not None:
        base.add_panel_label(ax, panel_label)


# --- Figures ---------------------------------------------------------------
def plot_sa_ra_scatter_grid(
    report: base.Report,
    output_dir: Path,
    figure_no: int,
    sample_size: int,
    formats: Sequence[str],
    train_smiles_path: Path,
    csv_paths: Dict[str, Path],
) -> None:
    """Fig9: 2x2 SA vs RA scatter, one panel per generation setting."""
    base.set_publication_style()
    train_smiles = load_train_smiles(train_smiles_path)

    panel_specs: List[Tuple[str, str, Path]] = [
        ("Unconditional generation", "Unconditional", csv_paths["Unconditional"]),
        ("Conditional (LogP = 1)", "LogP = 1", csv_paths["LogP = 1"]),
        ("Conditional (LogP = 2)", "LogP = 2", csv_paths["LogP = 2"]),
        ("Conditional (LogP = 3)", "LogP = 3", csv_paths["LogP = 3"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    panel_stats: List[SAScatterStats] = []
    for i, ((title, _group, csv_path), ax) in enumerate(zip(panel_specs, axes_flat)):
        stats = load_sa_ra_scatter(
            csv_path=csv_path,
            train_smiles=train_smiles,
            sample_size=sample_size,
            seed_offset=figure_no * 100 + i,
        )
        panel_stats.append(stats)
        _draw_scatter_panel(ax, stats, title=title, panel_label=PANEL_LABELS[i])

    fig.tight_layout(w_pad=1.2, h_pad=1.4)

    stem = f"Fig{figure_no}_SA_RA_scatter_2x2"
    base.save_figure(fig, output_dir, stem, formats=formats)

    figure_label = f"Fig. {figure_no}"
    for i, ((_title, group, csv_path), stats) in enumerate(zip(panel_specs, panel_stats)):
        rel_csv = csv_path.relative_to(PROJECT_ROOT).as_posix()
        note = (
            f"SA vs RA scatter; sampled up to {sample_size} valid rows; "
            f"identical-to-training={stats.in_train_n}, "
            f"de-novo-novel={stats.out_train_n}."
        )
        report.add_row(
            base.SummaryRow(
                figure=figure_label,
                panel=PANEL_LABELS[i],
                dataset="MOSES",
                group=group,
                source_file=rel_csv,
                raw_n=stats.raw_n,
                sampled_n=stats.sampled_n,
                valid_n=stats.valid_n,
                note=note,
            )
        )

    report.add_row(
        base.SummaryRow(
            figure=figure_label,
            panel="all",
            dataset="MOSES",
            group="Training reference (SMILES set)",
            source_file=train_smiles_path.relative_to(PROJECT_ROOT).as_posix(),
            raw_n=len(train_smiles),
            sampled_n=len(train_smiles),
            valid_n=len(train_smiles),
            note="Reference SMILES set used to classify generated molecules as identical-to-training.",
        )
    )

    report.add_caption(
        figure_label,
        "SA score versus RA score for MOSES molecules generated by the diffusion model under one unconditional and three "
        "conditional (target LogP = 1, 2 and 3) settings. Each panel shows up to "
        f"{sample_size} valid molecules sampled with a fixed random seed. Blue points denote de novo molecules that are "
        "absent from the training set, while red points denote generated SMILES that are identical to a training-set entry. "
        "Lower SA scores and higher RA scores indicate better synthetic accessibility and retro-synthesis feasibility, "
        "respectively; sample counts are reported in source_data_summary.csv.",
    )


def plot_sa_ra_kde_2panel(
    report: base.Report,
    output_dir: Path,
    figure_no: int,
    sample_size: int,
    formats: Sequence[str],
    bandwidth_scale: float,
    train_csv: Path,
    generation_csvs: Dict[str, Path],
) -> None:
    """Fig10: 2-panel KDE of SA (left) and RA (right) for training + 4 generations."""
    base.set_publication_style()

    # Group order defines both the drawing order (training on top) and the
    # legend order.
    groups: List[Tuple[str, Path, str, str, float]] = [
        ("Training set", train_csv, TRAIN_COLOR, "--", 1.2),
        ("Unconditional", generation_csvs["Unconditional"], GENERATION_COLORS["Unconditional"], "-", 1.35),
        ("LogP = 1", generation_csvs["LogP = 1"], GENERATION_COLORS["LogP = 1"], "-", 1.35),
        ("LogP = 2", generation_csvs["LogP = 2"], GENERATION_COLORS["LogP = 2"], "-", 1.35),
        ("LogP = 3", generation_csvs["LogP = 3"], GENERATION_COLORS["LogP = 3"], "-", 1.35),
    ]

    # Load SA + RA once per group, so both panels stay consistent.
    sa_data: List[Tuple[str, MetricValues, str, str, float]] = []
    ra_data: List[Tuple[str, MetricValues, str, str, float]] = []
    for i, (label, csv_path, color, linestyle, linewidth) in enumerate(groups):
        sa = load_metric_values(csv_path, "sa_score", sample_size, seed_offset=figure_no * 100 + i * 2)
        ra = load_metric_values(csv_path, "ra_score", sample_size, seed_offset=figure_no * 100 + i * 2 + 1)
        sa_data.append((label, sa, color, linestyle, linewidth))
        ra_data.append((label, ra, color, linestyle, linewidth))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))

    _draw_kde_panel(
        axes[0],
        sa_data,
        x_min=1.0,
        x_max=6.0,
        x_label="SA score (\u2193)",
        xticks=[1, 2, 3, 4, 5, 6],
        bandwidth_scale=bandwidth_scale,
        panel_label="a",
    )
    _draw_kde_panel(
        axes[1],
        ra_data,
        x_min=0.6,
        x_max=1.0,
        x_label="RA score (\u2191)",
        xticks=[0.6, 0.7, 0.8, 0.9, 1.0],
        bandwidth_scale=bandwidth_scale,
        panel_label="b",
    )

    fig.tight_layout(w_pad=1.6)

    stem = f"Fig{figure_no}_SA_RA_KDE_2panel"
    base.save_figure(fig, output_dir, stem, formats=formats)

    figure_label = f"Fig. {figure_no}"
    # Log per-panel provenance rows (SA and RA share the same source CSVs but
    # use independent sampling seeds; we record both for full traceability).
    for panel, panel_data, metric_name in [
        ("a", sa_data, "sa_score"),
        ("b", ra_data, "ra_score"),
    ]:
        for label, metric, _color, _ls, _lw in panel_data:
            csv_path = train_csv if label == "Training set" else generation_csvs[label]
            report.add_row(
                base.SummaryRow(
                    figure=figure_label,
                    panel=panel,
                    dataset="MOSES",
                    group=label,
                    source_file=csv_path.relative_to(PROJECT_ROOT).as_posix(),
                    raw_n=metric.raw_n,
                    sampled_n=metric.sampled_n,
                    valid_n=metric.valid_n,
                    note=f"{metric_name} KDE curve; target_valid_n={sample_size}; bandwidth_scale={bandwidth_scale:g}.",
                )
            )

    report.add_caption(
        figure_label,
        "Kernel density estimates of SA score (a) and RA score (b) for the MOSES training set and for molecules generated "
        "under one unconditional and three conditional (target LogP = 1, 2 and 3) settings. Curves were computed with a "
        "Gaussian KDE on up to "
        f"{sample_size} valid molecules sampled per group with a fixed random seed. Lower SA scores indicate easier "
        "synthesis; higher RA scores indicate greater retro-synthesis feasibility. Sample counts are reported in "
        "source_data_summary.csv.",
    )


# --- CLI --------------------------------------------------------------------
def _default_csv_paths() -> Tuple[Path, Dict[str, Path]]:
    train_csv = LZPNEED_DIR / "MOSEStrain.csv"
    generation_csvs = {
        "Unconditional": LZPNEED_DIR / "MOSESgen.csv",
        "LogP = 1": LZPNEED_DIR / "MOSESto1.csv",
        "LogP = 2": LZPNEED_DIR / "MOSESto2.csv",
        "LogP = 3": LZPNEED_DIR / "MOSESto3.csv",
    }
    return train_csv, generation_csvs


def parse_args() -> argparse.Namespace:
    train_csv_default, gen_csv_defaults = _default_csv_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Redraw the LZPneed SA/RA figures (Fig9 scatter grid and Fig10 KDE 2-panel) "
            "in the unified Nature-submission style."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported figure files (shared with the Fig1-Fig8 pipeline).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Target number of valid rows sampled per CSV; if fewer are available, all valid rows are used.",
    )
    parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=1.1,
        help="Multiplicative factor on Scott's KDE bandwidth (higher = smoother).",
    )
    parser.add_argument(
        "--train-smiles",
        type=Path,
        default=TRAIN_SMILES_PATH,
        help="Path to the training-set SMILES file used for Fig9 membership colouring.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=train_csv_default,
        help="Path to the MOSES training-set CSV used for Fig10 KDE curves.",
    )
    parser.add_argument("--uncond-csv", type=Path, default=gen_csv_defaults["Unconditional"])
    parser.add_argument("--logp1-csv", type=Path, default=gen_csv_defaults["LogP = 1"])
    parser.add_argument("--logp2-csv", type=Path, default=gen_csv_defaults["LogP = 2"])
    parser.add_argument("--logp3-csv", type=Path, default=gen_csv_defaults["LogP = 3"])
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "svg", "eps", "tiff", "jpg"],
        default=EXPORT_FORMATS,
        help="Export formats. Same choices as the Fig1-Fig8 pipeline.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=["append", "overwrite"],
        default="append",
        help=(
            "How to update figure_captions.txt / source_data_summary.csv. "
            "'append' keeps existing Fig1-Fig8 records; 'overwrite' rewrites both files with only Fig9/Fig10."
        ),
    )
    parser.add_argument(
        "--skip-scatter",
        action="store_true",
        help="Skip Fig9 (2x2 SA-RA scatter).",
    )
    parser.add_argument(
        "--skip-kde",
        action="store_true",
        help="Skip Fig10 (SA/RA KDE 2-panel).",
    )
    return parser.parse_args()


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
    """Append Fig9/Fig10 records to figure_captions.txt / source_data_summary.csv.

    In ``append`` mode existing rows for the same figures are removed first
    so that repeated runs stay idempotent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    caption_path = output_dir / "figure_captions.txt"
    csv_path = output_dir / "source_data_summary.csv"

    new_figures = {fig for fig, _ in report.captions}

    # --- captions -----------------------------------------------------------
    # Caption blocks in figure_captions.txt look like ``Fig. 9. <body>``; the
    # figure key itself contains a period, so we match it explicitly instead
    # of splitting on the first ``.``.
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
            caption_body = match.group(2).strip()
            if fig_key in new_figures:
                continue
            existing_captions.append((fig_key, caption_body))

    with caption_path.open("w", encoding="utf-8") as f:
        for fig_key, caption_body in existing_captions + report.captions:
            f.write(f"{fig_key}. {caption_body}\n\n")

    # --- source data summary -----------------------------------------------
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


# --- Entry point ------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_csvs: Dict[str, Path] = {
        "Unconditional": args.uncond_csv,
        "LogP = 1": args.logp1_csv,
        "LogP = 2": args.logp2_csv,
        "LogP = 3": args.logp3_csv,
    }

    # Validate inputs early with a clear error message.
    missing: List[Path] = []
    if not args.skip_scatter and not args.train_smiles.exists():
        missing.append(args.train_smiles)
    if not args.skip_kde and not args.train_csv.exists():
        missing.append(args.train_csv)
    for path in generation_csvs.values():
        if not path.exists():
            missing.append(path)
    if missing:
        joined = "\n  - ".join(str(p) for p in missing)
        raise SystemExit(f"The following required files are missing:\n  - {joined}")

    report = base.Report()

    if not args.skip_scatter:
        plot_sa_ra_scatter_grid(
            report=report,
            output_dir=output_dir,
            figure_no=9,
            sample_size=args.sample_size,
            formats=args.formats,
            train_smiles_path=args.train_smiles,
            csv_paths=generation_csvs,
        )
        print("Saved Fig9 SA-RA scatter grid.")

    if not args.skip_kde:
        plot_sa_ra_kde_2panel(
            report=report,
            output_dir=output_dir,
            figure_no=10,
            sample_size=args.sample_size,
            formats=args.formats,
            bandwidth_scale=args.bandwidth_scale,
            train_csv=args.train_csv,
            generation_csvs=generation_csvs,
        )
        print("Saved Fig10 SA/RA KDE 2-panel.")

    _write_summary(report, output_dir, mode=args.summary_mode)
    print(
        f"Updated {output_dir / 'figure_captions.txt'} and "
        f"{output_dir / 'source_data_summary.csv'} (mode={args.summary_mode})."
    )


if __name__ == "__main__":
    main()
