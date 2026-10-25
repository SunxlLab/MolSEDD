"""Token vocabularies for the QM9 / ZINC / MOSES SMILES datasets.

Every dataset is modelled as a character-level (atom-level) language model:

* ``0``                 padding
* ``1 .. N``            the N tokens of the dataset vocabulary
* ``N + 1``             end-of-sequence, rendered as ``$``

The vocabularies are **hard-coded on purpose**.  The released checkpoints were
trained with exactly these index orders (they were produced by sorting the token
set of each training file), so re-deriving them from a differently filtered
dataset would silently shift every embedding.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

PAD_ID = 0
END_CHAR = "$"

# QM9 (used by the 22-token QM9 checkpoints)
QM9_TOKENS = [
    "[NH3+]", "4", ")", "C", "#", "(", "[CH-]", "O", "[NH+]", "[NH2+]",
    "[N-]", "F", "=", "1", "N", "2", "5", "3", "[C-]", "[O-]",
]

# ZINC 250k (used by the 64-token ZINC checkpoints)
ZINC_TOKENS = [
    "c", "C", "(", ")", "1", "O", "2", "=", "N", "n", "3", "[C@@H]", "[C@H]",
    "F", "S", "Cl", "s", "-", "[NH+]", "o", "4", "/", "[O-]", "[nH]", "[NH2+]",
    "#", "Br", "[N+]", "[C@]", "[NH3+]", "[C@@]", "\\", "[nH+]", "5", "[N-]",
    "[S@]", "[S@@]", "I", "[n+]", "[S-]", "[n-]", "6", "P", "[P@@]", "[P@]",
    "[NH-]", "[o+]", "7", "[OH+]", "[S+]", "[CH-]", "[SH+]", "8", "[P@@H]",
    "[PH2]", "[P+]", "[CH2-]", "[O+]", "[s+]", "[PH+]", "[PH]", "[S@@+]",
]

# MOSES (used by the 28-token MOSES checkpoints)
MOSES_TOKENS = sorted([
    "#", "(", ")", "-", "1", "2", "3", "4", "5", "6", "=", "Br", "C", "Cl",
    "F", "I", "N", "O", "P", "S", "[H]", "[nH]", "c", "n", "o", "s",
])


@dataclass
class DatasetSpec:
    """Everything that has to agree between training and generation."""

    name: str
    tokens: List[str]
    seq_len: int
    min_len: int
    max_len: int
    start_tokens: Sequence[str] = field(default_factory=list)

    def __post_init__(self):
        self.tokens = list(self.tokens)
        self._encode = {tok: i + 1 for i, tok in enumerate(self.tokens)}
        self._encode[END_CHAR] = self.end_id
        self._decode = {i: tok for tok, i in self._encode.items()}
        self._decode[PAD_ID] = ""
        # Longest-first alternation so that ``[C@@H]`` wins over ``[C@]`` and
        # ``Br`` wins over ``B``.
        pattern = "|".join(map(re.escape, sorted(self._encode, key=len, reverse=True)))
        self._regex = re.compile(f"({pattern})")

    # ---------------------------------------------------------------- ids ----
    @property
    def end_id(self) -> int:
        return len(self.tokens) + 1

    @property
    def vocab_size(self) -> int:
        return len(self.tokens) + 2

    # ----------------------------------------------------------- encoding ----
    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into vocabulary tokens (unknown chars dropped)."""
        return self._regex.findall(smiles)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        ids = []
        for tok in tokens:
            if tok not in self._encode:
                raise ValueError(f"Token {tok!r} is not in the {self.name} vocabulary.")
            ids.append(self._encode[tok])
        return ids

    def to_ids(self, smiles: str, add_end: bool = True) -> List[int]:
        ids = self.encode(self.tokenize(smiles))
        return ids + [self.end_id] if add_end else ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode an id sequence, stopping at the first pad / end token."""
        out = []
        for i in ids:
            i = int(i)
            if i == PAD_ID or i == self.end_id:
                break
            out.append(self._decode.get(i, ""))
        return "".join(out)

    def id_to_char(self, i: int) -> str:
        return self._decode.get(int(i), "")


DATASETS: Dict[str, DatasetSpec] = {
    "qm9": DatasetSpec("qm9", QM9_TOKENS, seq_len=22, min_len=6, max_len=22,
                       start_tokens=["C", "N", "O", "F"]),
    "zinc": DatasetSpec("zinc", ZINC_TOKENS, seq_len=48, min_len=29, max_len=48,
                        start_tokens=["C", "c", "N", "O"]),
    "moses": DatasetSpec("moses", MOSES_TOKENS, seq_len=47, min_len=24, max_len=47,
                         start_tokens=["Br", "C", "Cl", "F", "N", "O", "[H]", "c"]),
}


def get_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    return DATASETS[key]
