"""分子侧的评分：Crippen logP，以及 MCTS 用的目标性质奖励。

这里只保留搜索过程本身需要的量。有效性/唯一性/新颖性等分布级指标不在本仓库计算。
"""

import contextlib
import math
from typing import Dict, Optional, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen


@contextlib.contextmanager
def quiet_rdkit():
    """暂时关掉 RDKit 的解析警告（生成结果里出现非法 SMILES 是正常的）。"""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(RDLogger.WARNING)


def mol_logp(smiles: str) -> Optional[float]:
    """Crippen logP；SMILES 解析不了时返回 ``None``。"""
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
    """R = exp(-MAE)：批次正好命中目标时为 1.0。"""
    return math.exp(-mae)


def property_scores(smiles_list: Sequence[str], target: float, min_validity: float = 0.5) -> Dict[str, float]:
    """对一批生成分子统计性质，给出 MCTS 用的奖励。

    ``efficiency`` 是 RDKit 能解析的比例；低于 ``min_validity`` 时整批视为失败
    （奖励记 0），与搜索过程中的判定一致。
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
