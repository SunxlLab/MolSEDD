"""Molecule-side metrics: validity / uniqueness / novelty and target-property scores."""

import contextlib
import math
from typing import Dict, Iterable, List, Optional, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen


@contextlib.contextmanager
def quiet_rdkit():
    """Temporarily silence RDKit parse warnings (invalid SMILES are expected here)."""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(RDLogger.WARNING)


def is_valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    with quiet_rdkit():
        return Chem.MolFromSmiles(smiles) is not None


def mol_logp(smiles: str) -> Optional[float]:
    """Crippen logP, or ``None`` if the SMILES does not parse."""
    with quiet_rdkit():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Crippen.MolLogP(mol)
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# Target-property reward used by the MCTS search
# --------------------------------------------------------------------------- #
def reward_from_mae(mae: float) -> float:
    """R = exp(-MAE) — 1.0 when the batch hits the target exactly."""
    return math.exp(-mae)


def property_scores(smiles_list: Sequence[str], target: float, min_validity: float = 0.5) -> Dict[str, float]:
    """Average / best property values over a generated batch.

    ``efficiency`` is the fraction of parseable molecules; below ``min_validity``
    the batch is treated as a failure (reward 0), exactly as in the search code.
    """
    logps = []
    best_smi, best_logp, min_diff = None, None, float("inf")
    with quiet_rdkit():
        for smi in smiles_list:
            logp = mol_logp(smi)
            if logp is None:
                continue
            logps.append(logp)
            diff = abs(target - logp)
            if diff < min_diff:
                min_diff, best_smi, best_logp = diff, smi, logp

    efficiency = len(logps) / len(smiles_list) if smiles_list else 0.0
    if efficiency > min_validity and logps:
        avg_logp = sum(logps) / len(logps)
        mae = sum(abs(target - x) for x in logps) / len(logps)
        reward = reward_from_mae(mae)
    else:
        avg_logp, mae, reward = -100.0, 100.0, 0.0

    return {
        "efficiency": efficiency,
        "avg_logp": avg_logp,
        "mae": mae,
        "reward": reward,
        "best_smiles": best_smi,
        "best_logp": best_logp,
    }


# --------------------------------------------------------------------------- #
# Distribution-level metrics
# --------------------------------------------------------------------------- #
def post_process_filter(smiles: str) -> bool:
    """Cheap syntax cleanup used before computing distribution metrics."""
    if not smiles:
        return False
    if smiles.count("(") != smiles.count(")"):
        return False
    if smiles.count("1") % 2 != 0:
        return False
    if smiles[0] in "#=" or smiles[-1] in "#=":
        return False
    return True


def distribution_metrics(
    generated: Iterable[str],
    train_smiles: Optional[Iterable[str]] = None,
    sample_size: int = 1000,
    apply_filter: bool = True,
) -> Dict[str, float]:
    """Validity / uniqueness / novelty of a list of generated SMILES."""
    gen = [s for s in generated if s]
    if apply_filter:
        gen = [s for s in gen if post_process_filter(s)]
    if sample_size and len(gen) > sample_size:
        gen = gen[:sample_size]

    valid = []
    with quiet_rdkit():
        for smi in gen:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid.append(Chem.MolToSmiles(mol))

    metrics = {
        "num_generated": float(len(gen)),
        "num_valid": float(len(valid)),
        "validity": len(valid) / len(gen) if gen else 0.0,
        "uniqueness": len(set(valid)) / len(valid) if valid else 0.0,
    }

    if train_smiles is not None:
        train_set = set()
        with quiet_rdkit():
            for smi in train_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    train_set.add(Chem.MolToSmiles(mol))
        unique_valid = set(valid)
        novel = unique_valid - train_set
        metrics["novelty"] = len(novel) / len(unique_valid) if unique_valid else 0.0
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["validity", "uniqueness", "novelty"]
    lines = [f"{k:>12}: {metrics[k]:.4f}" for k in keys if k in metrics]
    lines.append(f"{'generated':>12}: {int(metrics.get('num_generated', 0))}")
    lines.append(f"{'valid':>12}: {int(metrics.get('num_valid', 0))}")
    return "\n".join(lines)
