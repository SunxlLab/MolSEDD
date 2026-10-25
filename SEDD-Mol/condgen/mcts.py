"""Property-targeted MCTS over SMILES prefixes, guided by a pretrained SEDD model.

The search grows a SMILES prefix token by token.  At every node the SEDD model is
queried with the prefix clamped (see :class:`sedd_mol.generator.MolecularGenerator`)
to (a) estimate the node's property reward and (b) discover which next tokens the
model actually proposes.

Two selection rules are available:

* ``puct``  ``Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + n(s,a))``
* ``uct``   ``Q(s,a) + c * sqrt(2 * log N(s) / n(s,a))``

with ``P(s,a)`` either the empirical next-token frequency of the SEDD model
(``prior="sedd"``, the proposed method) or a uniform distribution
(``prior="uniform"``, the ablation).

The reward of a node is ``R = exp(-MAE)`` over a batch sampled with that prefix
clamped, and is set to ``0`` when the batch validity is below ``min_validity``.

All ablations reported in the paper are combinations of four flags, see
``VARIANTS`` in :mod:`condgen.run_mcts`.
"""

import csv
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from rdkit import Chem

from sedd_mol.metrics import property_scores, quiet_rdkit
from sedd_mol.vocab import DatasetSpec

_start_time = time.time()


def log_print(*args, **kwargs):
    """Timestamped stdout logging (the paper logs were produced this way)."""
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(ts, *args, **kwargs, flush=True)


@dataclass
class SearchConfig:
    # --- search rule -------------------------------------------------------
    selection: str = "puct"          # "puct" | "uct"
    prior: str = "sedd"              # "sedd" | "uniform"  (ignored for uct)
    dynamic_vocab: bool = True       # discover actions from the model vs. full vocab
    validity_filter: bool = True     # drop unparsable molecules when discovering actions
    # --- budget ------------------------------------------------------------
    simulations: int = 500
    c_param: float = 0.618
    discovery_bs: int = 100
    eval_bs: int = 100
    num_repeats: int = 1
    # --- objective ---------------------------------------------------------
    target: float = 1.0              # target Crippen logP
    min_validity: float = 0.5
    max_steps: int = 46
    seed: int = 42


class MCTSNode:
    __slots__ = (
        "tokens", "parent", "children", "visits", "value", "is_dead",
        "untried_actions", "actions_discovered", "prior", "action_priors",
    )

    def __init__(self, tokens, parent=None, prior=0.0):
        self.tokens: List[str] = tokens
        self.parent: Optional["MCTSNode"] = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.is_dead = False
        self.untried_actions: List[str] = []
        self.actions_discovered = False
        self.prior = prior
        self.action_priors: Dict[str, float] = {}

    def is_fully_expanded(self):
        return self.actions_discovered and not self.untried_actions

    def best_child(self, c_param: float, selection: str):
        valid = [c for c in self.children if not c.is_dead]
        if not valid:
            return None
        if selection == "puct":
            weights = [
                (c.value / c.visits) + c_param * c.prior * math.sqrt(self.visits) / (1 + c.visits)
                for c in valid
            ]
        else:
            weights = [
                (c.value / c.visits) + c_param * math.sqrt(2 * math.log(self.visits) / c.visits)
                for c in valid
            ]
        return valid[weights.index(max(weights))]


class MCTSSearch:
    """Property-targeted MCTS driven by a :class:`MolecularGenerator`."""

    def __init__(self, generator, spec: DatasetSpec, config: SearchConfig):
        self.gen = generator
        self.spec = spec
        self.cfg = config
        self._universe: List[str] = list(spec.tokens)
        self.log_rows: List[Dict] = []
        self.best = {
            "single_smiles": None, "single_logp": -100.0,
            "avg_prefix": None, "avg_logp": -100.0, "avg_reward": -1.0,
        }

    # ------------------------------------------------------------ helpers ----
    @staticmethod
    def _average(repeats: List[Dict[str, float]]) -> Dict[str, float]:
        n = max(len(repeats), 1)
        return {
            "reward": sum(r["reward"] for r in repeats) / n,
            "avg_logp": sum(r["avg_logp"] for r in repeats) / n,
            "mae": sum(r["mae"] for r in repeats) / n,
            "efficiency": sum(r["efficiency"] for r in repeats) / n,
            "samples": [round(r["reward"], 4) for r in repeats],
            "best_smiles": next((r["best_smiles"] for r in repeats if r["best_smiles"]), None),
            "best_logp": max(
                (r["best_logp"] for r in repeats if r["best_logp"] is not None), default=-100.0
            ),
        }

    def _evaluate(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Average property scores over ``num_repeats`` independent batches."""
        prefix = "".join(tokens)
        repeats = []
        for _ in range(self.cfg.num_repeats):
            try:
                smiles = self.gen.generate(prefix=prefix, batch_size=self.cfg.eval_bs)
            except Exception as exc:  # pragma: no cover - defensive
                log_print(f"Generation failed for prefix {prefix!r}: {exc}")
                smiles = []
            repeats.append(property_scores(smiles, self.cfg.target, self.cfg.min_validity))
        return self._average(repeats)

    def _discover(self, node: MCTSNode) -> Dict[str, float]:
        """Run one simulation on ``node``: score it and register its children."""
        prefix = "".join(node.tokens)
        depth = len(node.tokens)
        counter = Counter()

        if self.cfg.dynamic_vocab:
            # One batch is used for *both* the reward estimate and the next-token
            # statistics, exactly like the released experiment logs.
            repeats = []
            with quiet_rdkit():
                for _ in range(self.cfg.num_repeats):
                    try:
                        smiles = self.gen.generate(prefix=prefix, batch_size=self.cfg.discovery_bs)
                    except Exception as exc:  # pragma: no cover - defensive
                        log_print(f"Discovery failed for prefix {prefix!r}: {exc}")
                        smiles = []
                    for smi in smiles:
                        if self.cfg.validity_filter and Chem.MolFromSmiles(smi) is None:
                            continue
                        toks = self.spec.tokenize(smi)
                        if len(toks) <= depth or toks[:depth] != list(node.tokens):
                            continue
                        if toks[depth] in self._universe:
                            counter[toks[depth]] += 1
                    repeats.append(
                        property_scores(smiles, self.cfg.target, self.cfg.min_validity)
                    )
            scores = self._average(repeats)
        else:
            scores = self._evaluate(node.tokens)

        self._register_actions(node, counter)
        self._update_best(node, scores)

        prior_str = ""
        if self.cfg.prior == "sedd" and node.action_priors:
            top = sorted(node.action_priors.items(), key=lambda x: -x[1])[:5]
            prior_str = " | Priors: [" + ", ".join(f"{t}:{p:.3f}" for t, p in top) + (
                ", ...]" if len(node.action_priors) > 5 else "]"
            )
        log_print(
            f"    [Disc & Sim] Prefix: {prefix} | Eff: {scores['efficiency']:.2%} | "
            f"AvgLogP: {scores['avg_logp']:.4f} | MAE: {scores['mae']:.4f} | "
            f"R: {scores['reward']:.4f} | RSamples: {scores['samples']} | "
            f"Found {len(node.untried_actions)} children{prior_str}"
        )
        return scores

    def _register_actions(self, node: MCTSNode, counter: Counter):
        if self.cfg.dynamic_vocab:
            total = sum(counter.values())
            if self.cfg.prior == "sedd":
                node.action_priors = {t: c / total for t, c in counter.items()} if total else {}
            else:
                k = len(counter)
                node.action_priors = {t: 1.0 / k for t in counter} if k else {}
            node.untried_actions = sorted(counter.keys())
        else:
            node.untried_actions = list(self._universe)
        random.shuffle(node.untried_actions)
        node.actions_discovered = True

    def _update_best(self, node: MCTSNode, scores: Dict[str, float]):
        best_smi, best_logp = scores["best_smiles"], scores["best_logp"]
        if best_smi is not None and abs(self.cfg.target - best_logp) < abs(
            self.cfg.target - self.best["single_logp"]
        ):
            self.best["single_smiles"] = best_smi
            self.best["single_logp"] = best_logp
        if scores["avg_logp"] > -90 and scores["reward"] > self.best["avg_reward"]:
            self.best["avg_reward"] = scores["reward"]
            self.best["avg_logp"] = scores["avg_logp"]
            self.best["avg_prefix"] = "".join(node.tokens)

    def _backprop(self, node: MCTSNode, reward: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    # --------------------------------------------------------------- run ----
    def run(
        self,
        start_tokens: Optional[Sequence[str]] = None,
        root_prefix: Sequence[str] = (),
    ) -> Dict:
        cfg = self.cfg
        self._universe = list(self.spec.tokens)

        root = MCTSNode(list(root_prefix))
        if root_prefix or not cfg.dynamic_vocab:
            root.actions_discovered = False
        if start_tokens and not root_prefix:
            valid = [t for t in start_tokens if t in self._universe]
            if not valid:
                log_print(f"Warning: none of {start_tokens} is in the vocabulary; falling back to ['C'].")
                valid = ["C"]
            root.untried_actions = list(valid)
            random.shuffle(root.untried_actions)
            root.actions_discovered = True
            root.visits = 1
            root.action_priors = {t: 1.0 / len(valid) for t in valid}
        if not cfg.dynamic_vocab:
            # Static-vocabulary search: only the root is restricted to the start
            # tokens, every other node may try the full vocabulary.
            if not (start_tokens and not root_prefix):
                root.untried_actions = list(self._universe)
                random.shuffle(root.untried_actions)
            root.actions_discovered = True

        log_print(
            f"Running {cfg.selection.upper()} MCTS | prior={cfg.prior} | "
            f"dynamic_vocab={cfg.dynamic_vocab} | validity_filter={cfg.validity_filter} | "
            f"simulations={cfg.simulations} | c_param={cfg.c_param} | target_logP={cfg.target}"
        )

        for i in range(cfg.simulations):
            node = root

            # 1. selection ---------------------------------------------------
            while True:
                if len(node.tokens) >= cfg.max_steps:
                    break
                if not node.actions_discovered:
                    break
                if node.untried_actions:
                    break
                if not node.children:
                    node.is_dead = True
                    break
                nxt = node.best_child(cfg.c_param, cfg.selection)
                if nxt is None:
                    node.is_dead = True
                    break
                node = nxt
            if node.is_dead:
                continue

            # 2. expansion + simulation --------------------------------------
            if not node.actions_discovered and len(node.tokens) < cfg.max_steps:
                scores = self._discover(node)
            elif node.untried_actions and len(node.tokens) < cfg.max_steps:
                action = node.untried_actions.pop()
                prior = node.action_priors.get(action, 0.0)
                child = MCTSNode(node.tokens + [action], parent=node, prior=prior)
                node.children.append(child)
                scores = self._discover(child)
                node = child
            else:
                continue

            # 3. backpropagation -------------------------------------------
            reward = scores["reward"]
            dead = reward == 0.0 and (
                (scores["avg_logp"] <= -90) if cfg.dynamic_vocab else True
            )
            if dead:
                node.is_dead = True
                log_print(f"    [Dead Node] Prefix: {''.join(node.tokens)} died.")
            else:
                self._backprop(node, reward)

            self.log_rows.append(
                {
                    "simulation": i + 1,
                    "prefix": "".join(node.tokens),
                    "depth": len(node.tokens),
                    "efficiency": scores["efficiency"],
                    "avg_logp": scores["avg_logp"],
                    "mae": scores["mae"],
                    "reward": reward,
                    "best_so_far": self.best["avg_reward"],
                }
            )

            if (i + 1) % 10 == 0:
                log_print(
                    f"Loop {i + 1}/{cfg.simulations} | BestAvgReward: {self.best['avg_reward']:.4f} | "
                    f"BestAvgPrefix: {self.best['avg_prefix']} | Elapsed: {time.time() - _start_time:.0f}s"
                )

        return self._report(root)

    # ------------------------------------------------------------ results ----
    def _trace(self, root: MCTSNode, key) -> List[str]:
        node, tokens = root, []
        while node.children:
            valid = [c for c in node.children if c.visits > 0]
            if not valid:
                break
            child = max(valid, key=key)
            tokens.append(child.tokens[-1])
            log_print(
                f"  Step {len(tokens)}: {tokens[-1]} (Visits: {child.visits}/{node.visits}, "
                f"Value: {child.value:.2f}, Prior: {child.prior:.4f})"
            )
            node = child
        return tokens

    def _report(self, root: MCTSNode) -> Dict:
        log_print("\n=== MCTS completed ===")
        results = {}

        log_print("Best path by visit counts (robust child):")
        robust = self._trace(root, lambda c: c.visits)
        log_print(f"Robust sequence prefix: {''.join(robust)}")
        results["robust_prefix"] = "".join(robust)
        results["robust"] = self._evaluate(robust) if robust else {}

        log_print("\nBest path by mean reward:")
        by_value = self._trace(root, lambda c: c.value / c.visits)
        log_print(f"Max reward sequence prefix: {''.join(by_value)}")
        results["max_reward_prefix"] = "".join(by_value)
        results["max_reward"] = self._evaluate(by_value) if by_value else {}

        log_print("\nGlobal best found during search:")
        log_print(f"  [Single molecule] {self.best['single_smiles']} "
                  f"(logP={self.best['single_logp']:.4f})")
        log_print(f"  [Batch average]   {self.best['avg_prefix']} "
                  f"(avg logP={self.best['avg_logp']:.4f})")
        results.update(self.best)
        return results

    # ---------------------------------------------------------------- io ----
    def save_csv(self, path: str):
        if not self.log_rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.log_rows[0]))
            writer.writeheader()
            writer.writerows(self.log_rows)
        log_print(f"Search trajectory written to {path}")
