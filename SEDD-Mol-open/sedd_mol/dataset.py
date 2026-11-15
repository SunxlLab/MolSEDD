"""Dataset construction for SMILES corpora.

A dataset is stored as plain text (``.smi`` = one SMILES per line, optionally
followed by a name, or ``.csv`` with a ``SMILES`` column).  Every SMILES string
is tokenized with the vocabulary of :mod:`sedd_mol.vocab`, the end-of-sequence
token is appended, sequences are filtered by length and finally right-padded
with ``0`` to a fixed length.
"""

import os
from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from .vocab import DatasetSpec, get_spec


def load_smiles(path: str) -> List[str]:
    """Read a SMILES file and return the first whitespace-separated column."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} does not exist. Run `python tools/prepare_data.py --help` to build it."
        )
    if path.lower().endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path, encoding="utf-8")
        if "SMILES" not in df.columns:
            raise ValueError(f"{path!r} has no 'SMILES' column (found {list(df.columns)}).")
        return [s.strip() for s in df["SMILES"].dropna().astype(str)]

    smiles = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("smiles"):
                continue
            smiles.append(line.split()[0])
    return smiles


def encode_corpus(spec: DatasetSpec, smiles: List[str], min_len: int, max_len: int) -> List[List[int]]:
    """Tokenize + filter. ``min_len``/``max_len`` are inclusive and count the end token."""
    seqs = []
    for smi in smiles:
        ids = spec.to_ids(smi)
        if min_len <= len(ids) <= max_len:
            seqs.append(ids)
    return seqs


def build_dataset(
    smiles_path: str,
    dataset: str = "qm9",
    seq_len: Optional[int] = None,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    seed: int = 33,
):
    """Build a padded ``datasets.Dataset`` with a single ``input_ids`` column."""
    from datasets import Dataset

    spec = get_spec(dataset)
    seq_len = seq_len or spec.seq_len
    min_len = spec.min_len if min_len is None else min_len
    max_len = spec.max_len if max_len is None else max_len

    seqs = encode_corpus(spec, load_smiles(smiles_path), min_len, max_len)
    if not seqs:
        raise RuntimeError(
            f"No sequence left after filtering with {min_len} <= len <= {max_len}. "
            "Check --min_len/--max_len or the tokenizer vocabulary."
        )
    longest = max(len(s) for s in seqs)
    if longest > seq_len:
        raise RuntimeError(
            f"Longest sequence ({longest}) exceeds seq_len={seq_len}; "
            "increase data.seq_len or lower --max_len."
        )

    padded = [s + [0] * (seq_len - len(s)) for s in seqs]
    ds = Dataset.from_dict({"input_ids": padded})
    return ds.shuffle(seed=seed).with_format("torch")


def cycle_loader(dataloader, sampler=None):
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def get_dataloaders(config, distributed=True):
    """Training loader used by :mod:`sedd.run_train`."""
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Train Batch Size {config.training.batch_size} is not divisible by "
            f"config.ngpus={config.ngpus} gpus with accumulation {config.training.accum}."
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size {config.eval.batch_size} is not divisible by "
            f"config.ngpus={config.ngpus} gpus with accumulation {config.training.accum}."
        )

    train_set = build_dataset(
        smiles_path=config.data.path,
        dataset=config.data.name,
        seq_len=config.model.length,
        min_len=getattr(config.data, "min_len", None),
        max_len=getattr(config.data, "max_len", None),
        seed=getattr(config.data, "seed", 33),
    )

    train_sampler = DistributedSampler(train_set) if distributed else None
    train_loader = cycle_loader(
        DataLoader(
            train_set,
            batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=config.data.num_workers > 0,
            drop_last=True,
        )
    )
    return train_loader
