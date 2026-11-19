# MolSEDD

[中文](README.md) | [English](README_en.md)

A molecule-generation codebase built on **score-entropy discrete diffusion (SEDD)**. It
covers model training on QM9 / ZINC-250k / MOSES, unconditional generation, conditional
generation, and the accompanying search and comparative-experiment scripts.

This README only documents the repository layout and how to run each part; it does not
describe the implementation.

---

## 1. Repository layout

```
.
├── train.py                     training entry point (Hydra)
│
├── configs/
│   ├── config.yaml              QM9 + uniform (default)
│   ├── config_zinc.yaml         ZINC-250k
│   ├── config_moses.yaml        MOSES
│   ├── config_qm9_absorb.yaml   QM9 + absorbing
│   └── model/                   DDiT backbone sizes (small / medium)
│
├── data/                        data shipped with the repository (see section 3)
│   ├── conditional/             conditional samples (MOSES, 4 × 1000)
│   └── unconditional/           unconditional samples (MOSES, 10000)
│   # training sets are not included; build them into data/train/ (see section 3.3)
│
├── generate/                    unconditional generation
│   ├── sample_qm9.py
│   ├── sample_zinc.py
│   └── sample_moses.py
│
├── conditional/                 conditional generation (fixed SMILES prefix)
│   ├── cond_qm9.py
│   ├── cond_zinc.py
│   └── cond_moses.py
│
├── mcts/                        property-targeted search
│   ├── search.py                search engine
│   ├── run_qm9.py
│   ├── run_zinc.py
│   └── run_moses.py
│
├── ablation/                    comparative experiments
│   ├── run_ablation.py
│   └── run_ablation.sh
│
├── sedd/                        discrete diffusion core
│   ├── graph_lib.py             uniform / absorbing forward process
│   ├── noise_lib.py             geometric / log-linear noise schedules
│   ├── sampling.py              predictors, denoiser, sampler
│   ├── catsample.py             categorical sampling
│   ├── losses.py                training objective, optimizer, train step
│   ├── load_model.py            load from an experiment directory / HuggingFace hub
│   ├── run_train.py             DDP training worker (spawned by train.py)
│   ├── utils.py                 logging and checkpoints
│   └── model/                   Transformer backbone + EMA
│
├── sedd_mol/                    molecule layer
│   ├── vocab.py                 vocabularies and sequence lengths of the three datasets
│   ├── dataset.py               SMILES → token ids → DataLoader
│   ├── generator.py             sampling wrapper (supports a fixed prefix)
│   └── metrics.py               validity / uniqueness / novelty, target-property scores
│
├── tools/prepare_data.py        rebuild the training corpus (rarely needed)
├── environment.yml
├── requirements.txt
└── LICENSE
```

`sedd/` and `sedd_mol/` are the shared foundation; `generate/`, `conditional/`, `mcts/`
and `ablation/` are their callers. Every script can be run on its own.

---

## 2. Dataset specifications

The vocabulary and sequence length are pinned in `sedd_mol/vocab.py`, in
`configs/config_*.yaml` and in the `SPEC` block at the top of each generation script.
**All three places must match the checkpoint you load**, otherwise the token ids shift.

| Dataset | Vocabulary | Sequence length | Training length filter |
|---|---|---|---|
| QM9 | 20 SMILES tokens + pad + end = 22 | 22 | 6 – 22 |
| ZINC-250k | 62 SMILES tokens + pad + end = 64 | 48 | 29 – 48 |
| MOSES | 26 SMILES tokens + pad + end = 28 | 47 | 24 – 47 |

---

## 3. Data

The repository ships only the MOSES sample sets (conditional and unconditional) so that the
evaluation pipeline can be reproduced out of the box. **Training sets are not included**;
prepare them as described in section 3.3 and place them under `data/train/`.

### 3.1 `data/conditional/` — conditional samples (MOSES)

| File | Molecules | Target |
|---|---|---|
| `MOSESto1_n1000.txt` | 1,000 | logP ≈ 1 |
| `MOSESto2_n1000.txt` | 1,000 | logP ≈ 2 |
| `MOSESto3_n1000.txt` | 1,000 | logP ≈ 3 |
| `MOSESto4_n1000.txt` | 1,000 | logP ≈ 4 |

### 3.2 `data/unconditional/` — unconditional samples (MOSES)

| File | Molecules |
|---|---|
| `MOSES.txt` | 10,000 |

All these files contain one SMILES per line and no labels. Use `sedd_mol/metrics.py` for
validity / uniqueness / novelty; the training set is the reference set for novelty.

### 3.3 Training sets (download them yourself)

The training sets are not redistributed here because of the licences of the original
datasets. QM9, ZINC-250k and MOSES are all public — please download them from the
respective original papers (or their official project pages), and put the cleaned SMILES
files under `data/train/` (which is where `data.path` in `configs/*.yaml` points by
default):

| Dataset | File to prepare |
|---|---|
| QM9 | `data/train/trainQM9.txt` |
| ZINC-250k | `data/train/trainZINC.txt` |
| MOSES | `data/train/trainMOSES.txt` |

Keep the raw downloads in `data/raw/` (git-ignored), then convert them with
`tools/prepare_data.py`, which applies the length filter of section 2 and prints the token
statistics:

```bash
python tools/prepare_data.py --dataset moses --input data/raw/<your raw file> \
    --output data/train/trainMOSES.txt --stats
```

Conditional and unconditional samples for QM9 / ZINC are not included either; generate them
with the scripts in sections 6 and 7.

---

## 4. Installation

```bash
conda env create -f environment.yml      # creates the "sedd" environment (CUDA 11.8)
conda activate sedd
```

Or, on top of an existing PyTorch installation:

```bash
pip install -r requirements.txt
```

`flash-attn` is **optional**: if it cannot be imported the model falls back to
`torch.nn.functional.scaled_dot_product_attention` (`sedd/model/transformer.py`).

---

## 5. Training

Make sure `data/train/train{QM9,ZINC,MOSES}.txt` exists first (not shipped, see section 3.3).

```bash
python train.py                                   # QM9 + uniform (default)
python train.py --config-name config_zinc         # ZINC-250k
python train.py --config-name config_moses        # MOSES
python train.py --config-name config_qm9_absorb   # QM9 + absorbing

# multi-GPU / larger batch
python train.py --config-name config_moses ngpus=4 training.batch_size=1024
```

Each run is written to `exp_local/<dataset>/<date>/<time>/`:

```
exp_local/moses/2026.11.19/120000/
├── .hydra/config.yaml                 # required by load_model
├── checkpoints/checkpoint_*.pth
├── checkpoints-meta/checkpoint.pth    # resuming / loading
├── samples/
├── logs
└── config.json
```

All hyper-parameters live in `configs/config_*.yaml` and can be overridden on the command
line with `field=value` (e.g. `noise.type=geometric graph.type=absorb model=small`).

---

## 6. Unconditional generation

One script per dataset; the sequence length and vocabulary are defined in the `SPEC` block
at the top of each file:

```bash
python generate/sample_qm9.py   --model_path exp_local/qm9/2026.11.19/120000 \
    --num_samples 10000 --batch_size 500 --steps 1024

python generate/sample_zinc.py  --model_path exp_local/zinc/2026.11.19/120000 \
    --num_samples 10000 --batch_size 500 --steps 1024

python generate/sample_moses.py --model_path exp_local/moses/2026.11.19/120000 \
    --num_samples 10000 --batch_size 500 --steps 1024
```

Output defaults to `samples/<dataset>_<timestamp>.txt`, one SMILES per line; use `--output`
to change the path. `--num_samples` defaults to 10000 and `--steps` to 1024.

---

## 7. Conditional generation (fixed prefix)

Give a SMILES prefix and let the model complete the rest:

```bash
python conditional/cond_qm9.py   --model_path exp_local/qm9/2026.11.19/120000 \
    --prefix "FOCC1=C" --num_samples 1000 --wanti 1.0

python conditional/cond_zinc.py  --model_path exp_local/zinc/2026.11.19/120000 \
    --prefix "CC" --num_samples 1000

python conditional/cond_moses.py --model_path exp_local/moses/2026.11.19/120000 \
    --prefix "CC" --num_samples 1000
```

Output defaults to `samples/cond_<dataset>_<timestamp>.txt`. The script prints the prefix
retention rate; with `--wanti` (target logP) it additionally reports the mean logP of the
batch, its deviation, and the molecule closest to the target.

---

## 8. Property-targeted search

`mcts/` searches for SMILES prefixes with respect to a target property; one entry point per
dataset:

```bash
python mcts/run_qm9.py   --model_path exp_local/qm9/2026.11.19/120000   --wanti 1.0
python mcts/run_zinc.py  --model_path exp_local/zinc/2026.11.19/120000  --wanti 1.0
python mcts/run_moses.py --model_path exp_local/moses/2026.11.19/120000 --wanti 1.0
```

The default search configuration for each dataset lives in `DATASET_CONFIG` inside the
corresponding script and can be overridden on the command line (`--simulations`, `--c_param`,
`--discovery_bs`, `--num_repeats`, `--seed`, ...).

Output is written to `mcts_out/`; every run produces a pair of files:

* `<dataset>_..._<timestamp>.csv` — one row per simulation with the property statistics of
  that step and the running best, ready for plotting convergence curves;
* `<same name>.json` — arguments, runtime and the evaluation of the final candidate paths.

A discovered prefix can be fed directly to `conditional/` to generate full molecules.

---

## 9. Comparative experiments

`ablation/` reproduces the comparative experiments with different search configurations:

```bash
# run all four configurations (default)
python ablation/run_ablation.py --dataset moses \
    --model_path exp_local/moses/2026.11.19/120000

# run a single one
python ablation/run_ablation.py --dataset moses --model_path ... --variants puct

# shell wrapper
MODEL_PATH=exp_local/moses/2026.11.19/120000 DATASET=moses ./ablation/run_ablation.sh
```

Available configuration names: `puct` / `uct` / `uct_novalid` / `pure`; see `VARIANTS` in
`ablation/run_ablation.py`. Results are written to `mcts_out/ablation/`.

---

## 10. Notes

1. **Model and data specs must match.** When loading a checkpoint, the vocabulary size and
   the sequence length must be the same as during training; a `size mismatch` error usually
   means the dataset was changed without changing the config.
2. **`model_path` points to an experiment directory**, not to a `.pth` file. The directory
   must contain `.hydra/config.yaml` and `checkpoints-meta/checkpoint.pth` (or
   `checkpoints/checkpoint_*.pth`). A HuggingFace repo id is accepted as well.
3. **Out of memory**: lower `--batch_size` on the generation side, or lower
   `training.batch_size` / raise `training.accum` on the training side.
4. **Training sets must be downloaded.** `data/train/*.txt` is not in the repository (the
   MOSES training set is about 56 MB); prepare it as in section 3.3. `data/raw/` is already
   git-ignored.
5. **`data/` only ships the MOSES samples.** QM9 / ZINC samples are not included; generate
   them with the scripts in sections 6 and 7.

---

## 11. Citation and licence

If you find this code useful, please cite our paper (bibtex to be added) as well as the
original SEDD work:

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

MIT — see [LICENSE](LICENSE). The code under `sedd/` derives from
[SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) (MIT).
