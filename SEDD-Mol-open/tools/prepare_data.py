"""Build the training text files used by ``train.py``.

The tokenized corpora are *not* shipped with the repository (QM9 alone is 215 MB).
Download the raw datasets, then run e.g.::

    # QM9  (https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip)
    python tools/prepare_data.py --dataset qm9 --input qm9.csv --output data/qm9_train.txt

    # ZINC 250k (already a .smi file)
    python tools/prepare_data.py --dataset zinc --input 250k_rndm_zinc_drugs_clean.smi \
        --output data/zinc_train.txt

    # MOSES (train split of https://github.com/molecularsets/moses)
    python tools/prepare_data.py --dataset moses --input train.txt --output data/moses_train.txt

Use ``--stats`` to print the token statistics of a corpus (this is how the
vocabularies in ``sedd_mol/vocab.py`` were originally obtained).
"""

import argparse
import os
from collections import Counter

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedd_mol.dataset import encode_corpus, load_smiles  # noqa: E402
from sedd_mol.vocab import get_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Prepare SMILES training files")
    parser.add_argument("--dataset", type=str, required=True, choices=["qm9", "zinc", "moses"])
    parser.add_argument("--input", type=str, required=True, help="Raw .smi / .txt / .csv file")
    parser.add_argument("--output", type=str, default=None, help="Where to write the cleaned SMILES")
    parser.add_argument("--min_len", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N molecules")
    parser.add_argument("--stats", action="store_true", help="Print token / length statistics")
    parser.add_argument("--write_token_list", type=str, default=None,
                        help="Optional path for the sorted token list (see sedd_mol/vocab.py)")
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    min_len = spec.min_len if args.min_len is None else args.min_len
    max_len = spec.max_len if args.max_len is None else args.max_len

    smiles = load_smiles(args.input)
    print(f"Read {len(smiles)} SMILES from {args.input}")

    kept, dropped_unknown = [], 0
    for smi in smiles:
        ids = spec.to_ids(smi)
        if not ids[:-1]:
            dropped_unknown += 1
            continue
        if min_len <= len(ids) <= max_len:
            kept.append(smi)
        if args.limit and len(kept) >= args.limit:
            break

    print(f"Kept {len(kept)} / {len(smiles)} after length filter "
          f"({min_len} <= len <= {max_len}); {dropped_unknown} dropped (no known token)")

    if args.stats:
        token_counter = Counter()
        for smi in kept:
            token_counter.update(spec.tokenize(smi))
        lengths = [len(spec.to_ids(s)) for s in kept]
        print(f"Distinct tokens: {len(token_counter)} | vocab size (incl. pad/end): {spec.vocab_size}")
        print("Top tokens:", token_counter.most_common(15))
        print(f"Length: min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.1f}")
        unused = [t for t in spec.tokens if t not in token_counter]
        if unused:
            print("Vocabulary tokens not present in this corpus:", unused)

    if args.write_token_list:
        tokens = sorted({t for smi in kept for t in spec.tokenize(smi)})
        with open(args.write_token_list, "w", encoding="utf-8") as f:
            f.write("\n".join(tokens))
        print(f"Token list written to {args.write_token_list} ({len(tokens)} tokens)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.writelines(s + "\n" for s in kept)
        print(f"Wrote {len(kept)} SMILES to {args.output}")

    seqs = encode_corpus(spec, kept, min_len, max_len)
    longest = max(len(s) for s in seqs)
    print(f"Longest tokenized sequence: {longest} (config data.seq_len should be >= {longest})")


if __name__ == "__main__":
    main()
