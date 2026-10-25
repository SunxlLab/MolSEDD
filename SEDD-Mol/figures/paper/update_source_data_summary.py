from __future__ import annotations

import csv
from pathlib import Path

import redraw_nature_figures as figs

ROOT = figs.PROJECT_ROOT
OUTPUT = ROOT / "natureimg" / "images" / "source_data_summary.csv"
TARGET_VALID_N = 3000


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def count_logp(path: Path, sample_size: int, seed_offset: int = 0):
    return figs.logp_values_from_file(path, sample_size, seed_offset=seed_offset)


def count_metrics(path: Path, sample_size: int, seed_offset: int = 0):
    return figs.metrics_from_file(path, sample_size, seed_offset=seed_offset)


def count_fingerprints(path: Path, sample_size: int, seed_offset: int):
    _, stats = figs.sampled_fingerprints_from_file(path, sample_size, seed_offset)
    return stats


def count_target_molecules(
    target: float,
):
    selected = figs.molecules_from_user_selection()[target]
    return figs.ValuesWithStats(
        values=figs.np.empty((0,), dtype=float),
        raw_n=len(figs.USER_SELECTED_MOSES_TARGET_SMILES[target]),
        sampled_n=len(figs.USER_SELECTED_MOSES_TARGET_SMILES[target]),
        valid_n=len(selected),
    )


def add_row(rows, figure, panel, dataset, group, path, stats, note):
    rows.append(
        {
            "figure": figure,
            "panel": panel,
            "dataset": dataset,
            "group": group,
            "source_file": rel(path) if isinstance(path, Path) else path,
            "raw_n": stats.raw_n,
            "sampled_n": stats.sampled_n,
            "valid_n": stats.valid_n,
            "note": note,
        }
    )


def main() -> None:
    rows = []

    # Fig. 1 and Fig. 2 target 3000 valid RDKit-parsed molecules per group, unless fewer are available.
    fig1_sources = [
        ("Training set", ROOT / "train" / "trainQM9.txt"),
        ("Target LogP = -1", ROOT / "cond" / "QM9to-1.txt"),
        ("Target LogP = 0", ROOT / "cond" / "QM9to0.txt"),
        ("Target LogP = 1", ROOT / "cond" / "QM9to1.txt"),
    ]
    for i, (group, path) in enumerate(fig1_sources):
        add_row(rows, "Fig. 1", "a", "QM9", group, path, count_logp(path, TARGET_VALID_N, 100 + i), f"LogP KDE; target_valid_n={TARGET_VALID_N}.")

    fig2_sources = [
        ("Training set", ROOT / "train" / "trainMOSES.txt"),
        ("Target LogP = 1", ROOT / "cond" / "MOSESto1.txt"),
        ("Target LogP = 2", ROOT / "cond" / "MOSESto2.txt"),
        ("Target LogP = 3", ROOT / "cond" / "MOSESto3.txt"),
    ]
    for i, (group, path) in enumerate(fig2_sources):
        add_row(rows, "Fig. 2", "a", "MOSES", group, path, count_logp(path, TARGET_VALID_N, 200 + i), f"LogP KDE; target_valid_n={TARGET_VALID_N}.")

    # Fig. 3 targets 3000 valid RDKit-parsed molecules per group, unless fewer are available.
    fig3_sources = [
        ("all", "Training set", ROOT / "train" / "trainQM9.txt"),
        ("a", "Step 0", ROOT / "condtsne" / "qm9to-1_0.txt"),
        ("b", "Step 500", ROOT / "condtsne" / "qm9to-1_500.txt"),
        ("c", "Step 1000", ROOT / "condtsne" / "qm9to-1_1000.txt"),
        ("d", "Step 2000", ROOT / "condtsne" / "qm9to-1_2000.txt"),
    ]
    for i, (panel, group, path) in enumerate(fig3_sources):
        add_row(rows, "Fig. 3", panel, "QM9", group, path, count_logp(path, TARGET_VALID_N, 300 + i), f"LogP evolution KDE; target_valid_n={TARGET_VALID_N}.")

    # Fig. 4, Fig. 5 and Fig. 6 target 3000 valid RDKit-parsed molecules per group, unless fewer are available.
    metric_sources = [
        ("Fig. 4", "QM9", ROOT / "train" / "trainQM9.txt", ROOT / "generate" / "QM9.txt", 400),
        ("Fig. 5", "ZINC", ROOT / "train" / "trainZINC.txt", ROOT / "generate" / "ZINC.txt", 500),
        ("Fig. 6", "MOSES", ROOT / "train" / "trainMOSES.txt", ROOT / "generate" / "MOSES.txt", 600),
    ]
    for figure, dataset, train_path, gen_path, seed_base in metric_sources:
        add_row(rows, figure, "all", dataset, "Training set", train_path, count_metrics(train_path, TARGET_VALID_N, seed_base), f"LogP/QED/TPSA/MolWt KDE; target_valid_n={TARGET_VALID_N}.")
        add_row(rows, figure, "all", dataset, "Generated", gen_path, count_metrics(gen_path, TARGET_VALID_N, seed_base + 1), f"LogP/QED/TPSA/MolWt KDE; target_valid_n={TARGET_VALID_N}.")

    # Fig. 7 targets 3000 valid molecules per train/generated group, unless fewer are available.
    tsne_sources = [
        ("a", "QM9", "Training set", ROOT / "train" / "1.txt", 700),
        ("a", "QM9", "Generated", ROOT / "generate" / "QM9.txt", 701),
        ("b", "ZINC", "Training set", ROOT / "train" / "trainZINC.txt", 710),
        ("b", "ZINC", "Generated", ROOT / "generate" / "ZINC.txt", 711),
        ("c", "MOSES", "Training set", ROOT / "train" / "trainMOSES.txt", 720),
        ("c", "MOSES", "Generated", ROOT / "generate" / "MOSES.txt", 721),
    ]
    for panel, dataset, group, path, seed_offset in tsne_sources:
        add_row(rows, "Fig. 7", panel, dataset, group, path, count_fingerprints(path, TARGET_VALID_N, seed_offset), f"Morgan fingerprint t-SNE; dataset-wise PCA/t-SNE; target_valid_n={TARGET_VALID_N}; random_seed=42.")

    # Fig. 8 shows user-specified molecular structures for each target LogP.
    for panel, target in [("a", 1.0), ("b", 2.0), ("c", 3.0)]:
        add_row(
            rows,
            "Fig. 8",
            panel,
            "MOSES",
            f"Target LogP = {target:g}",
            "user-specified SMILES in task file",
            count_target_molecules(target),
            f"User-specified molecular structures for target LogP {target:g}; RDKit-calculated LogP values are shown above each molecule.",
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["figure", "panel", "dataset", "group", "source_file", "raw_n", "sampled_n", "valid_n", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {OUTPUT}")


if __name__ == "__main__":
    main()
