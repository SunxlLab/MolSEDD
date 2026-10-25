from __future__ import annotations

import argparse
import csv
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw, QED
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting/chemistry dependency: "
        f"{exc.name}\n\n"
        "Install the required packages in the Python environment used to run this script, for example:\n"
        "python -m pip install numpy matplotlib scipy scikit-learn pillow rdkit\n"
    ) from exc

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "natureimg" / "images"
SEED = 42
EXPORT_FORMATS = ["pdf", "svg", "eps", "tiff", "jpg"]
FORMAT_SUBDIRS = {fmt: fmt for fmt in EXPORT_FORMATS}
DPI = 600


TRAIN_COLOR = "#666666"
TRAIN_FILL = "#E6E6E6"
TEXT_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
TARGET_LINE_COLOR = "#4D4D4D"

DATASET_COLORS = {
    "QM9": "#1F5AA6",
    "ZINC": "#C45A11",
    "MOSES": "#8E2C2C",
}

CONDITION_COLORS = ["#9ECAE1", "#4292C6", "#08519C"]
EVOLUTION_COLORS = ["#B6D7EA", "#6BAED6", "#3182BD", "#08519C"]
PANEL_LABELS = list("abcdefghijklmnopqrstuvwxyz")


@dataclass
class SummaryRow:
    figure: str
    panel: str
    dataset: str
    group: str
    source_file: str
    raw_n: int
    sampled_n: int
    valid_n: int
    note: str = ""


class Report:
    def __init__(self) -> None:
        self.captions: List[Tuple[str, str]] = []
        self.rows: List[SummaryRow] = []

    def add_caption(self, figure: str, caption: str) -> None:
        self.captions.append((figure, " ".join(caption.split())))

    def add_row(self, row: SummaryRow) -> None:
        self.rows.append(row)

    def write(self, output_dir: Path) -> None:
        caption_path = output_dir / "figure_captions.txt"
        with caption_path.open("w", encoding="utf-8") as f:
            for fig, caption in self.captions:
                f.write(f"{fig}. {caption}\n\n")

        csv_path = output_dir / "source_data_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "figure",
                    "panel",
                    "dataset",
                    "group",
                    "source_file",
                    "raw_n",
                    "sampled_n",
                    "valid_n",
                    "note",
                ],
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.__dict__)


@dataclass
class ValuesWithStats:
    values: np.ndarray
    raw_n: int
    sampled_n: int
    valid_n: int


@dataclass
class MetricsWithStats:
    metrics: Dict[str, np.ndarray]
    raw_n: int
    sampled_n: int
    valid_n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw existing molecular-generation figures in a unified submission style."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported figure files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Target number of valid RDKit-parsed molecules for descriptor-distribution panels; if fewer are available, all valid molecules are used.",
    )
    parser.add_argument(
        "--evolution-sample-size",
        type=int,
        default=3000,
        help="Target number of valid RDKit-parsed molecules for LogP-evolution panels; if fewer are available, all valid molecules are used.",
    )
    parser.add_argument(
        "--tsne-sample-size",
        type=int,
        default=3000,
        help="Target number of valid train/generated molecules per dataset in the t-SNE panel; if fewer are available, all valid molecules are used.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--molecules-per-target",
        type=int,
        default=5,
        help="Number of user-specified MOSES molecules shown for each target LogP.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "svg", "eps", "tiff", "jpg"],
        default=EXPORT_FORMATS,
        help="Export formats. The target journal accepts TIFF, EPS and JPEG; PDF/SVG are retained for editing.",
    )
    parser.add_argument("--skip-tsne", action="store_true", help="Skip the t-SNE figure.")
    parser.add_argument("--skip-molecules", action="store_true", help="Skip the molecular-structure plate.")
    return parser.parse_args()


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


def read_smiles(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Source SMILES file does not exist: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def deterministic_candidate_order(items: Sequence[str], seed_offset: int = 0) -> List[str]:
    candidates = list(items)
    if len(candidates) > 1:
        rng = random.Random(SEED + seed_offset)
        rng.shuffle(candidates)
    return candidates


def deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def logp_values_from_file(path: Path, sample_size: int, seed_offset: int = 0) -> ValuesWithStats:
    smiles = read_smiles(path)
    values: List[float] = []
    sampled_n = 0
    for smi in deterministic_candidate_order(smiles, seed_offset=seed_offset):
        sampled_n += 1
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            values.append(float(Descriptors.MolLogP(mol)))
        except Exception:
            continue
        if len(values) >= sample_size:
            break
    arr = np.asarray(values, dtype=float)
    return ValuesWithStats(arr, raw_n=len(smiles), sampled_n=sampled_n, valid_n=len(arr))


def metrics_from_file(path: Path, sample_size: int, seed_offset: int = 0) -> MetricsWithStats:
    smiles = read_smiles(path)
    metrics = {
        "logp": [],
        "qed": [],
        "tpsa": [],
        "molwt": [],
    }
    valid_n = 0
    sampled_n = 0
    for smi in deterministic_candidate_order(smiles, seed_offset=seed_offset):
        sampled_n += 1
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            metrics["logp"].append(float(Descriptors.MolLogP(mol)))
            metrics["qed"].append(float(QED.qed(mol)))
            metrics["tpsa"].append(float(Descriptors.TPSA(mol)))
            metrics["molwt"].append(float(Descriptors.MolWt(mol)))
            valid_n += 1
        except Exception:
            continue
        if valid_n >= sample_size:
            break
    return MetricsWithStats(
        metrics={key: np.asarray(value, dtype=float) for key, value in metrics.items()},
        raw_n=len(smiles),
        sampled_n=sampled_n,
        valid_n=valid_n,
    )


def padded_range(arrays: Sequence[np.ndarray], pad_fraction: float = 0.08) -> Tuple[float, float]:
    merged = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if len(arr) > 0])
    if len(merged) == 0:
        return 0.0, 1.0
    low = float(np.min(merged))
    high = float(np.max(merged))
    if np.isclose(low, high):
        return low - 1.0, high + 1.0
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def plot_kde_line(
    ax,
    values: np.ndarray,
    label: str,
    color: str,
    x_range: np.ndarray,
    linestyle: str = "-",
    linewidth: float = 1.3,
    fill: bool = False,
    fill_color: str | None = None,
) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    try:
        kde = gaussian_kde(values)
        y = kde(x_range)
    except Exception:
        return 0.0
    ax.plot(x_range, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    if fill:
        ax.fill_between(x_range, y, 0, color=fill_color or color, alpha=1.0, linewidth=0)
    return float(np.nanmax(y)) if len(y) else 0.0


def style_density_axis(ax, xlabel: str, ylabel: bool = True) -> None:
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Probability density")
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle="-", zorder=0)
    ax.tick_params(axis="both", pad=1.5)


def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str] | None = None) -> None:
    ensure_output_dir(output_dir)
    formats = formats or EXPORT_FORMATS
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
    plt.close(fig)


def cleanup_legacy_root_exports(output_dir: Path, formats: Sequence[str]) -> None:
    """Remove old root-level Fig* exports after switching to format-specific folders."""
    for fmt in formats:
        for path in output_dir.glob(f"Fig*.{fmt}"):
            if path.is_file():
                path.unlink()


def plot_conditional_logp(
    report: Report,
    output_dir: Path,
    figure_no: int,
    dataset: str,
    train_path: Path,
    condition_paths: Sequence[Path],
    targets: Sequence[float],
    sample_size: int,
    panel_label: Optional[str] = "a",
) -> None:
    set_publication_style()
    train = logp_values_from_file(train_path, sample_size, seed_offset=figure_no * 100)
    cond_stats = [
        logp_values_from_file(path, sample_size, seed_offset=figure_no * 100 + i + 1)
        for i, path in enumerate(condition_paths)
    ]

    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    x_min, x_max = padded_range([train.values] + [item.values for item in cond_stats])
    if dataset == "MOSES":
        x_max = max(x_max, 8.0)
    x_range = np.linspace(x_min, x_max, 500)

    plot_kde_line(
        ax,
        train.values,
        "Training set",
        TRAIN_COLOR,
        x_range,
        linestyle="--",
        linewidth=1.2,
        fill=True,
        fill_color=TRAIN_FILL,
    )

    for color, target, item in zip(CONDITION_COLORS, targets, cond_stats):
        plot_kde_line(
            ax,
            item.values,
            f"Target LogP = {target:g}",
            color,
            x_range,
            linewidth=1.4,
            fill=False,
        )
        ax.axvline(target, color=color, linestyle=":", linewidth=0.8)

    ax.set_title(f"{dataset} conditional LogP")
    style_density_axis(ax, "LogP", ylabel=True)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", handlelength=2.0, borderaxespad=0.2)
    if panel_label is not None:
        add_panel_label(ax, panel_label)

    stem = f"Fig{figure_no}_conditional_logp_{dataset}"
    save_figure(fig, output_dir, stem)

    figure_label = f"Fig. {figure_no}"
    panel_record = panel_label if panel_label is not None else ""
    report.add_row(
        SummaryRow(
            figure=figure_label,
            panel=panel_record,
            dataset=dataset,
            group="Training set",
            source_file=str(train_path.relative_to(PROJECT_ROOT)),
            raw_n=train.raw_n,
            sampled_n=train.sampled_n,
            valid_n=train.valid_n,
            note="Grey dashed KDE curve.",
        )
    )
    for target, path, item in zip(targets, condition_paths, cond_stats):
        report.add_row(
            SummaryRow(
                figure=figure_label,
                panel=panel_record,
                dataset=dataset,
                group=f"Target LogP = {target:g}",
                source_file=str(path.relative_to(PROJECT_ROOT)),
                raw_n=item.raw_n,
                sampled_n=item.sampled_n,
                valid_n=item.valid_n,
                note="Colored solid KDE curve; dotted vertical line marks the target LogP.",
            )
        )

    target_text = ", ".join(f"{target:g}" for target in targets)
    report.add_caption(
        figure_label,
        f"Conditional LogP distributions for {dataset}. The grey dashed curve denotes the training-set reference; "
        f"colored solid curves denote generated molecules conditioned on target LogP values {target_text}. "
        "Dotted vertical lines mark the corresponding target values. Kernel density estimates were computed from valid RDKit-parsed molecules; "
        "sample counts are provided in source_data_summary.csv.",
    )


def plot_logp_evolution(
    report: Report,
    output_dir: Path,
    figure_no: int,
    train_path: Path,
    step_paths: Dict[str, Path],
    target: float,
    sample_size: int,
) -> None:
    set_publication_style()
    train = logp_values_from_file(train_path, sample_size, seed_offset=figure_no * 100)
    steps = {
        name: logp_values_from_file(path, sample_size, seed_offset=figure_no * 100 + i + 1)
        for i, (name, path) in enumerate(step_paths.items())
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.25), sharex=True, sharey=True)
    axes_flat = np.ravel(axes)
    x_min, x_max = padded_range([train.values] + [item.values for item in steps.values()])
    x_range = np.linspace(x_min, x_max, 500)

    step_titles = {
        "Step 0": "Initial",
        "Step 500": "Early",
        "Step 1000": "Refining",
        "Step 2000": "Converged",
    }

    y_max_values: List[float] = []
    for i, (ax, (step_name, item), color) in enumerate(zip(axes_flat, steps.items(), EVOLUTION_COLORS)):
        train_peak = plot_kde_line(
            ax,
            train.values,
            "Training set",
            TRAIN_COLOR,
            x_range,
            linestyle="--",
            linewidth=1.0,
            fill=True,
            fill_color=TRAIN_FILL,
        )
        generated_peak = plot_kde_line(
            ax,
            item.values,
            step_name,
            color,
            x_range,
            linewidth=1.35,
            fill=False,
        )
        ax.axvline(target, color=TARGET_LINE_COLOR, linestyle=":", linewidth=0.8)
        ax.set_title(f"{step_titles.get(step_name, step_name)}\n{step_name}")
        style_density_axis(ax, "LogP", ylabel=i == 0)
        y_max_values.extend([train_peak, generated_peak])
        add_panel_label(ax, PANEL_LABELS[i], x=-0.20, y=1.10)

    if y_max_values:
        shared_ymax = max(y_max_values) * 1.18
        for ax in axes_flat:
            ax.set_ylim(0, shared_ymax)

    handles = [
        Line2D([0], [0], color=TRAIN_COLOR, linestyle="--", linewidth=1.0, label="Training set"),
        Line2D([0], [0], color=EVOLUTION_COLORS[-1], linestyle="-", linewidth=1.35, label="Generated step"),
        Line2D([0], [0], color=TARGET_LINE_COLOR, linestyle=":", linewidth=0.8, label=f"Target LogP = {target:g}"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    stem = f"Fig{figure_no}_logp_evolution"
    save_figure(fig, output_dir, stem)

    figure_label = f"Fig. {figure_no}"
    report.add_row(
        SummaryRow(
            figure=figure_label,
            panel="all",
            dataset="QM9",
            group="Training set",
            source_file=str(train_path.relative_to(PROJECT_ROOT)),
            raw_n=train.raw_n,
            sampled_n=train.sampled_n,
            valid_n=train.valid_n,
            note="Shared grey dashed reference across all panels.",
        )
    )
    for i, (step_name, path) in enumerate(step_paths.items()):
        item = steps[step_name]
        report.add_row(
            SummaryRow(
                figure=figure_label,
                panel=PANEL_LABELS[i],
                dataset="QM9",
                group=step_name,
                source_file=str(path.relative_to(PROJECT_ROOT)),
                raw_n=item.raw_n,
                sampled_n=item.sampled_n,
                valid_n=item.valid_n,
                note=f"Generated molecules at {step_name}; vertical dotted line marks target LogP {target:g}.",
            )
        )

    report.add_caption(
        figure_label,
        f"Evolution of generated QM9 LogP distributions toward the target value {target:g}. Panels show the generated distribution at "
        "successive search steps overlaid on the training-set reference. The dotted vertical line marks the target LogP. "
        "Kernel density estimates were computed from valid RDKit-parsed molecules; sample counts are provided in source_data_summary.csv.",
    )


def plot_metrics_grid(
    report: Report,
    output_dir: Path,
    figure_no: int,
    dataset: str,
    train_path: Path,
    gen_path: Path,
    sample_size: int,
) -> None:
    set_publication_style()
    train = metrics_from_file(train_path, sample_size, seed_offset=figure_no * 100)
    gen = metrics_from_file(gen_path, sample_size, seed_offset=figure_no * 100 + 1)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharey=False)
    axes_flat = axes.ravel()
    metric_config = [
        ("logp", "LogP", "LogP"),
        ("qed", "QED", "QED"),
        ("tpsa", "TPSA", "TPSA (Å$^2$)"),
        ("molwt", "Molecular weight", "Molecular weight (g mol$^{-1}$)"),
    ]
    gen_color = DATASET_COLORS[dataset]

    for i, (ax, (key, _title, xlabel)) in enumerate(zip(axes_flat, metric_config)):
        x_min, x_max = padded_range([train.metrics[key], gen.metrics[key]])
        if key == "qed":
            x_min, x_max = 0.0, 1.0
        x_range = np.linspace(x_min, x_max, 500)
        plot_kde_line(
            ax,
            train.metrics[key],
            "Training set",
            TRAIN_COLOR,
            x_range,
            linestyle="--",
            linewidth=1.0,
            fill=True,
            fill_color=TRAIN_FILL,
        )
        plot_kde_line(
            ax,
            gen.metrics[key],
            "Generated",
            gen_color,
            x_range,
            linewidth=1.35,
            fill=False,
        )
        style_density_axis(ax, xlabel, ylabel=i in (0, 2))
        ax.set_ylim(bottom=0)
        add_panel_label(ax, PANEL_LABELS[i])

    handles = [
        Line2D([0], [0], color=TRAIN_COLOR, linestyle="--", linewidth=1.0, label="Training set"),
        Line2D([0], [0], color=gen_color, linestyle="-", linewidth=1.35, label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{dataset} molecular-property distributions", y=0.995, fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.925))

    stem = f"Fig{figure_no}_metrics_{dataset.lower()}"
    save_figure(fig, output_dir, stem)

    figure_label = f"Fig. {figure_no}"
    report.add_row(
        SummaryRow(
            figure=figure_label,
            panel="all",
            dataset=dataset,
            group="Training set",
            source_file=str(train_path.relative_to(PROJECT_ROOT)),
            raw_n=train.raw_n,
            sampled_n=train.sampled_n,
            valid_n=train.valid_n,
            note="Descriptors: LogP, QED, TPSA and molecular weight.",
        )
    )
    report.add_row(
        SummaryRow(
            figure=figure_label,
            panel="all",
            dataset=dataset,
            group="Generated",
            source_file=str(gen_path.relative_to(PROJECT_ROOT)),
            raw_n=gen.raw_n,
            sampled_n=gen.sampled_n,
            valid_n=gen.valid_n,
            note="Descriptors: LogP, QED, TPSA and molecular weight.",
        )
    )
    report.add_caption(
        figure_label,
        f"Comparison of generated and training-set molecular-property distributions for {dataset}. Panels show LogP, QED, TPSA and molecular weight. "
        "Grey dashed curves denote the training-set reference, and colored solid curves denote generated molecules. "
        "Kernel density estimates were computed from valid RDKit-parsed molecules; sample counts are provided in source_data_summary.csv.",
    )


def fingerprints_from_smiles(smiles: Sequence[str], n_bits: int = 2048) -> np.ndarray:
    fps: List[np.ndarray] = []
    for smi in smiles:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            continue
    if not fps:
        return np.empty((0, n_bits), dtype=np.int8)
    return np.vstack(fps)


def sampled_fingerprints_from_file(
    path: Path,
    sample_size: int,
    seed_offset: int,
) -> Tuple[np.ndarray, ValuesWithStats]:
    smiles = read_smiles(path)
    fps: List[np.ndarray] = []
    sampled_n = 0
    for smi in deterministic_candidate_order(smiles, seed_offset=seed_offset):
        sampled_n += 1
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            continue
        if len(fps) >= sample_size:
            break
    fp_array = np.vstack(fps) if fps else np.empty((0, 2048), dtype=np.int8)
    stats = ValuesWithStats(
        values=np.empty((0,), dtype=float),
        raw_n=len(smiles),
        sampled_n=sampled_n,
        valid_n=len(fp_array),
    )
    return fp_array, stats


def plot_tsne_three_data(
    report: Report,
    output_dir: Path,
    figure_no: int,
    sample_size: int,
    perplexity: float,
) -> None:
    set_publication_style()
    datasets = {
        "QM9": {
            "train": PROJECT_ROOT / "train" / "1.txt",
            "gen": PROJECT_ROOT / "generate" / "QM9.txt",
            "color": DATASET_COLORS["QM9"],
        },
        "ZINC": {
            "train": PROJECT_ROOT / "train" / "trainZINC.txt",
            "gen": PROJECT_ROOT / "generate" / "ZINC.txt",
            "color": DATASET_COLORS["ZINC"],
        },
        "MOSES": {
            "train": PROJECT_ROOT / "train" / "trainMOSES.txt",
            "gen": PROJECT_ROOT / "generate" / "MOSES.txt",
            "color": DATASET_COLORS["MOSES"],
        },
    }

    stats: Dict[Tuple[str, str], ValuesWithStats] = {}
    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    n_components_by_dataset: Dict[str, int] = {}

    for i, (name, config) in enumerate(datasets.items()):
        train_fps, train_stats = sampled_fingerprints_from_file(
            config["train"], sample_size=sample_size, seed_offset=figure_no * 100 + i * 10
        )
        gen_fps, gen_stats = sampled_fingerprints_from_file(
            config["gen"], sample_size=sample_size, seed_offset=figure_no * 100 + i * 10 + 1
        )
        stats[(name, "train")] = train_stats
        stats[(name, "gen")] = gen_stats

        if len(train_fps) == 0 and len(gen_fps) == 0:
            continue

        blocks = [block for block in [train_fps, gen_fps] if len(block) > 0]
        x = np.vstack(blocks)
        n_components = max(2, min(50, x.shape[0] - 1, x.shape[1]))
        n_components_by_dataset[name] = n_components
        x_pca = PCA(n_components=n_components, random_state=SEED).fit_transform(x)
        effective_perplexity = min(perplexity, max(5.0, (x.shape[0] - 1) / 3.0))
        x_tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=SEED,
            init="pca",
            learning_rate="auto",
        ).fit_transform(x_pca)

        embeddings[name] = {}
        current = 0
        if len(train_fps) > 0:
            embeddings[name]["train"] = x_tsne[current : current + len(train_fps)]
            current += len(train_fps)
        if len(gen_fps) > 0:
            embeddings[name]["gen"] = x_tsne[current : current + len(gen_fps)]

    if not embeddings:
        raise RuntimeError("No valid fingerprints were generated for t-SNE.")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=False, sharey=False)
    train_scatter_color = "#B8B8B8"

    for i, (ax, (name, config)) in enumerate(zip(axes, datasets.items())):
        train_data = embeddings.get(name, {}).get("train")
        gen_data = embeddings.get(name, {}).get("gen")
        if train_data is not None:
            ax.scatter(
                train_data[:, 0],
                train_data[:, 1],
                s=1.4,
                c=train_scatter_color,
                edgecolors="none",
                rasterized=True,
                label="Training set",
                zorder=1,
            )
        if gen_data is not None:
            ax.scatter(
                gen_data[:, 0],
                gen_data[:, 1],
                s=1.8,
                c=config["color"],
                edgecolors="none",
                rasterized=True,
                label="Generated",
                zorder=2,
            )
        ax.set_title(name)
        ax.set_xlabel("t-SNE 1")
        if i == 0:
            ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        add_panel_label(ax, PANEL_LABELS[i], x=-0.12, y=1.07)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=train_scatter_color, markeredgecolor="none", markersize=4, label="Training set"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#333333", markeredgecolor="none", markersize=4, label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    stem = f"Fig{figure_no}_tsne_3data"
    save_figure(fig, output_dir, stem)

    figure_label = f"Fig. {figure_no}"
    for i, (name, config) in enumerate(datasets.items()):
        for group in ["train", "gen"]:
            item = stats[(name, group)]
            report.add_row(
                SummaryRow(
                    figure=figure_label,
                    panel=PANEL_LABELS[i],
                    dataset=name,
                    group="Training set" if group == "train" else "Generated",
                    source_file=str(config[group].relative_to(PROJECT_ROOT)),
                    raw_n=item.raw_n,
                    sampled_n=item.sampled_n,
                    valid_n=item.valid_n,
                    note=f"Morgan fingerprints with radius 2 and 2048 bits; dataset-wise PCA to {n_components_by_dataset.get(name, 0)} dimensions; dataset-wise t-SNE perplexity {perplexity:g}; random seed {SEED}.",
                )
            )
    report.add_caption(
        figure_label,
        "t-SNE visualization of training and generated molecules across QM9, ZINC and MOSES. Grey points denote training molecules, "
        "and colored points denote generated molecules. Molecular fingerprints were computed as Morgan fingerprints with radius 2 and 2048 bits, "
        f"followed by dataset-wise PCA and t-SNE with perplexity {perplexity:g}. Sample counts and source files are provided in source_data_summary.csv.",
    )


def mol_image(mol, size: Tuple[int, int] = (900, 650)):
    return Draw.MolToImage(mol, size=size, kekulize=True, fitImage=True)


USER_SELECTED_MOSES_TARGET_SMILES: Dict[float, List[str]] = {
    1.0: [
        "NOc1nnccc1CN(C)C(=O)c1ccccc1",
        "NOc1nnccc1CNC(=O)C1Cc2cc(Cl)ccc2O1",
        "NOc1nnccc1CNC(=O)Cn1ccc2ccccc21",
        "NOc1nnccc1CNc1ccc2nc3n(c(=O)c2c1)CCC3",
        "NOc1nnccc1CNS(=O)(=O)c1c(F)cccc1Cl",
    ],
    2.0: [
        "OOc1cnc(C)cc1Cn1cnc2ccccc2c1=O",
        "OOc1cnc(OCCCC(=O)NC(C)CC(=O)N2CCCCC2)cc1",
        "OOc1cnc2c(c1)CCNC2CC1NC(=O)c2cccc(C)c2N1",
        "OOc1cnc(-c2nc3ccc(Cl)cn3c2C(N)=O)cc1",
        "OOc1cnc(SCC(=O)NC(C#N)c2ccco2)cc1",
    ],
    3.0: [
        "BrOc1cc(O)c(C(=O)NC(C(N)=O)c2cccc(Cl)c2)cc1C",
        "BrOc1cc(O)cc(C(=O)COC(=O)c2ccc(C#N)cc2)c1OC",
        "BrOc1cc(O)c2c(c1)C(C(O)c1ccccc1)NCC2",
        "BrOc1cc(O)ccc1C#CCNC(=O)c1ccccc1F",
        "BrOc1cc(O)c(NC(=O)NCc2nccs2)cc1F",
    ],
}


def molecules_from_user_selection() -> Dict[float, List[Tuple[str, object, float]]]:
    selected: Dict[float, List[Tuple[str, object, float]]] = {}
    invalid: List[str] = []
    for target, smiles_list in USER_SELECTED_MOSES_TARGET_SMILES.items():
        selected[target] = []
        for smi in smiles_list:
            mol = mol_from_smiles(smi)
            if mol is None:
                invalid.append(smi)
                continue
            selected[target].append((smi, mol, float(Descriptors.MolLogP(mol))))
    if invalid:
        joined = "\n".join(invalid)
        raise SystemExit(f"Invalid user-specified SMILES for Fig8:\n{joined}")
    return selected


def plot_moses_target_molecules(
    report: Report,
    output_dir: Path,
    figure_no: int,
    molecules_per_target: int,
    sample_size: int,
) -> None:
    set_publication_style()
    selected_rows = molecules_from_user_selection()
    targets = list(selected_rows.keys())
    rows = [selected_rows[target][:molecules_per_target] for target in targets]

    fig, axes = plt.subplots(
        len(targets),
        molecules_per_target,
        figsize=(7.2, 4.75),
        squeeze=False,
    )

    for row_idx, (target, ranked) in enumerate(zip(targets, rows)):
        for col_idx in range(molecules_per_target):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx == 0:
                ax.text(
                    -0.24,
                    0.5,
                    f"Target\nLogP = {target:g}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )
                add_panel_label(ax, PANEL_LABELS[row_idx], x=-0.30, y=1.05)
            if col_idx >= len(ranked):
                continue
            smi, mol, logp = ranked[col_idx]
            img = mol_image(mol)
            ax.imshow(img, interpolation="lanczos")
            ax.set_facecolor("white")
            ax.set_title(f"#{col_idx + 1}  LogP={logp:.2f}", fontsize=6.5, pad=1.5)

    fig.tight_layout(w_pad=0.2, h_pad=0.65)

    stem = f"Fig{figure_no}_MOSES_target_molecules"
    save_figure(fig, output_dir, stem)

    figure_label = f"Fig. {figure_no}"
    for row_idx, (target, ranked) in enumerate(zip(targets, rows)):
        report.add_row(
            SummaryRow(
                figure=figure_label,
                panel=PANEL_LABELS[row_idx],
                dataset="MOSES",
                group=f"Target LogP = {target:g}",
                source_file="user-specified SMILES in task file",
                raw_n=len(USER_SELECTED_MOSES_TARGET_SMILES[target]),
                sampled_n=len(USER_SELECTED_MOSES_TARGET_SMILES[target]),
                valid_n=len(ranked),
                note=f"User-specified molecular structures for target LogP {target:g}; RDKit-calculated LogP values are shown above each molecule.",
            )
        )

    report.add_caption(
        figure_label,
        f"User-specified representative MOSES-like molecules for target LogP values 1, 2 and 3. Each row shows {molecules_per_target} molecular structures provided for the corresponding target. "
        "Values above each structure show RDKit-calculated LogP.",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)
    global EXPORT_FORMATS
    EXPORT_FORMATS = list(args.formats)
    cleanup_legacy_root_exports(output_dir, EXPORT_FORMATS)
    report = Report()

    def run_step(label: str, func) -> None:
        print(f"[natureimg] Start {label}", flush=True)
        func()
        print(f"[natureimg] Finished {label}", flush=True)

    run_step(
        "Fig. 1 conditional LogP for QM9",
        lambda: plot_conditional_logp(
            report=report,
            output_dir=output_dir,
            figure_no=1,
            dataset="QM9",
            train_path=PROJECT_ROOT / "train" / "trainQM9.txt",
            condition_paths=[
                PROJECT_ROOT / "cond" / "QM9to-1.txt",
                PROJECT_ROOT / "cond" / "QM9to0.txt",
                PROJECT_ROOT / "cond" / "QM9to1.txt",
            ],
            targets=[-1.0, 0.0, 1.0],
            sample_size=args.sample_size,
        ),
    )

    run_step(
        "Fig. 2 conditional LogP for MOSES",
        lambda: plot_conditional_logp(
            report=report,
            output_dir=output_dir,
            figure_no=2,
            dataset="MOSES",
            train_path=PROJECT_ROOT / "train" / "trainMOSES.txt",
            condition_paths=[
                PROJECT_ROOT / "cond" / "MOSESto1.txt",
                PROJECT_ROOT / "cond" / "MOSESto2.txt",
                PROJECT_ROOT / "cond" / "MOSESto3.txt",
            ],
            targets=[1.0, 2.0, 3.0],
            sample_size=args.sample_size,
            panel_label=None,
        ),
    )

    run_step(
        "Fig. 3 LogP evolution",
        lambda: plot_logp_evolution(
            report=report,
            output_dir=output_dir,
            figure_no=3,
            train_path=PROJECT_ROOT / "train" / "trainQM9.txt",
            step_paths={
                "Step 0": PROJECT_ROOT / "condtsne" / "qm9to-1_0.txt",
                "Step 500": PROJECT_ROOT / "condtsne" / "qm9to-1_500.txt",
                "Step 1000": PROJECT_ROOT / "condtsne" / "qm9to-1_1000.txt",
                "Step 2000": PROJECT_ROOT / "condtsne" / "qm9to-1_2000.txt",
            },
            target=-1.0,
            sample_size=args.evolution_sample_size,
        ),
    )

    run_step(
        "Fig. 4 QM9 molecular-property distributions",
        lambda: plot_metrics_grid(
            report=report,
            output_dir=output_dir,
            figure_no=4,
            dataset="QM9",
            train_path=PROJECT_ROOT / "train" / "trainQM9.txt",
            gen_path=PROJECT_ROOT / "generate" / "QM9.txt",
            sample_size=args.sample_size,
        ),
    )

    run_step(
        "Fig. 5 ZINC molecular-property distributions",
        lambda: plot_metrics_grid(
            report=report,
            output_dir=output_dir,
            figure_no=5,
            dataset="ZINC",
            train_path=PROJECT_ROOT / "train" / "trainZINC.txt",
            gen_path=PROJECT_ROOT / "generate" / "ZINC.txt",
            sample_size=args.sample_size,
        ),
    )

    run_step(
        "Fig. 6 MOSES molecular-property distributions",
        lambda: plot_metrics_grid(
            report=report,
            output_dir=output_dir,
            figure_no=6,
            dataset="MOSES",
            train_path=PROJECT_ROOT / "train" / "trainMOSES.txt",
            gen_path=PROJECT_ROOT / "generate" / "MOSES.txt",
            sample_size=args.sample_size,
        ),
    )

    if not args.skip_tsne:
        run_step(
            "Fig. 7 t-SNE chemical-space visualization",
            lambda: plot_tsne_three_data(
                report=report,
                output_dir=output_dir,
                figure_no=7,
                sample_size=args.tsne_sample_size,
                perplexity=args.tsne_perplexity,
            ),
        )

    if not args.skip_molecules:
        run_step(
            "Fig. 8 MOSES target molecules",
            lambda: plot_moses_target_molecules(
                report=report,
                output_dir=output_dir,
                figure_no=8,
                molecules_per_target=args.molecules_per_target,
                sample_size=args.sample_size,
            ),
        )

    report.write(output_dir)
    format_dirs = ", ".join(str(output_dir / FORMAT_SUBDIRS.get(fmt, fmt)) for fmt in EXPORT_FORMATS)
    print(f"[natureimg] Finished. Submission-style figures were exported by format to: {format_dirs}", flush=True)


if __name__ == "__main__":
    main()
