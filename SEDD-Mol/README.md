# SEDD-Mol: Score-Entropy Discrete Diffusion for Molecule Generation and Property-Targeted Conditional Generation

This repository contains the official code for our work on **score-entropy discrete
diffusion (SEDD) for molecular generation**, including

1. **training** of character/atom-level SEDD models on QM9, ZINC-250k and MOSES,
2. **unconditional generation**,
3. **prefix/suffix constrained generation** (projection-based conditioning),
4. **property-targeted conditional generation** with a **SEDD-prior MCTS**
   (PUCT using the model's own next-token distribution as prior) together with all
   the baselines and ablations reported in the paper.

The framework builds on the SEDD codebase by Lou et al. (see
[Acknowledgements](#acknowledgements)).

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

---

## 1. Method in one paragraph

A molecule is a SMILES string, i.e. a sequence of discrete tokens. A SEDD model
learns the ratio of the data distribution at every noise level and is trained with
the score-entropy objective, which gives a *ratio-based* score function directly on
the token simplex (no continuous relaxation). Sampling is a τ-leaping style
Euler/analytic reverse process (`sedd/sampling.py`).

For **conditional generation** we keep the pretrained model completely frozen and
search over SMILES prefixes with MCTS. A node is a prefix; clamping the prefix at
every denoising step (projection) lets the model *complete* it, which gives

* a **reward** `R = exp(-MAE)` of the sampled property (e.g. Crippen logP) w.r.t. the
  target, and
* the **next-token statistics** `P(s, a)` of the model, used as the PUCT prior.

Selection is `Q(s,a) + c · P(s,a) · √N(s) / (1 + n(s,a))` for the proposed method, and
`Q(s,a) + c · √(2 ln N(s) / n(s,a))` (UCT) for the baselines. See
[`docs/mcts_formulas.md`](docs/mcts_formulas.md) for the full derivation.

---

## 2. Repository layout

```
.
├── train.py                  # training entry point (Hydra)
├── sample.py                 # unconditional generation
├── sample_cond.py            # prefix / suffix constrained generation
├── configs/                  # Hydra configs (qm9 / qm9-absorb / zinc / moses)
│   └── model/                # DiT/DDiT backbone sizes (small, medium, *_un)
├── sedd/                     # core discrete-diffusion framework
│   ├── graph_lib.py          #   uniform / absorbing forward process
│   ├── noise_lib.py          #   geometric / loglinear noise schedules
│   ├── sampling.py           #   predictors, denoiser, sampler
│   ├── catsample.py          #   categorical sampling
│   ├── losses.py             #   score-entropy loss, optimizer, train step
│   ├── utils.py              #   logging / checkpointing
│   ├── load_model.py         #   load from HF hub or a local experiment dir
│   ├── run_train.py          #   DDP training worker (spawned by train.py)
│   └── model/                #   transformer backbone (DDiT) + EMA
├── sedd_mol/                 # molecule layer
│   ├── vocab.py              #   QM9 / ZINC / MOSES vocabularies (frozen index order)
│   ├── dataset.py            #   SMILES -> padded token ids, dataloaders
│   ├── generator.py          #   prefix/suffix projection sampler
│   └── metrics.py            #   validity / uniqueness / novelty, logP reward
├── condgen/                  # property-targeted conditional generation
│   ├── mcts.py               #   unified MCTS engine (selection/prior/vocab/filter)
│   ├── run_mcts.py           #   CLI with all paper variants
│   └── run_baselines.py      #   greedy / beam / beam+MCTS
├── tools/
│   ├── prepare_data.py       #   build the training text files
│   ├── eval_metrics.py       #   distribution metrics of a sample file
│   └── benchmark_speed.py    #   sampling throughput sweep
├── scripts/
│   └── run_ablation.sh       #   runs all MCTS variants sequentially
├── figures/                  # plotting scripts (ablations, speed, loss)
│   └── paper/                # camera-ready figure scripts of the paper
├── docs/                     # method notes and hyper-parameters
└── results/                  # search logs behind the paper's figures
```

---

## 3. Installation

```bash
conda env create -f environment.yml      # creates the "sedd" env (CUDA 11.8)
conda activate sedd
```

or, with an existing PyTorch installation:

```bash
pip install -r requirements.txt
```

`flash-attn` is **optional**: if it is not importable, attention falls back to
`torch.nn.functional.scaled_dot_product_attention` automatically
(`sedd/model/transformer.py`).

---

## 4. Data preparation

Tokenized corpora are not distributed with the code (QM9 alone is 215 MB).
Download the raw datasets and convert them:

```bash
# QM9 — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
python tools/prepare_data.py --dataset qm9  --input qm9.csv --output data/qm9_train.txt --stats

# ZINC 250k — https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean.smi
python tools/prepare_data.py --dataset zinc --input 250k_rndm_zinc_drugs_clean.smi \
    --output data/zinc_train.txt --stats

# MOSES — https://github.com/molecularsets/moses  (train split)
python tools/prepare_data.py --dataset moses --input train.txt --output data/moses_train.txt --stats
```

The tokenizer vocabularies live in `sedd_mol/vocab.py` and are **frozen**: the token
index order must match the released checkpoints, so do not regenerate them from a
different corpus unless you retrain from scratch. `--stats` prints the token and
length statistics that were used to define them.

Sequence lengths (including the end token) and vocabulary sizes:

| dataset | tokens | end id | `model.length` | `tokens` | length filter |
|---------|--------|--------|----------------|----------|---------------|
| QM9     | 20     | 21     | 22             | 22       | 6 – 22        |
| ZINC    | 62     | 63     | 48             | 64       | 29 – 48       |
| MOSES   | 26     | 27     | 47             | 28       | 24 – 47       |

---

## 5. Training

```bash
python train.py                                   # QM9, uniform graph (default)
python train.py --config-name config_qm9_absorb    # QM9, absorbing graph
python train.py --config-name config_zinc          # ZINC 250k
python train.py --config-name config_moses         # MOSES
python train.py --config-name config_moses ngpus=4 training.batch_size=1024
```

Hydra writes every run to `exp_local/<dataset>/<date>/<time>/`:

```
exp_local/moses/2026.02.26/104218/
├── .hydra/config.yaml        # needed by load_model
├── checkpoints/checkpoint_*.pth
├── checkpoints-meta/checkpoint.pth   # used for resuming and by load_model
├── samples/
├── logs
└── config.json
```

Override any field on the command line, e.g.
`noise.type=geometric graph.type=absorb model=small`.

---

## 6. Unconditional generation

```bash
python sample.py --model_path exp_local/moses/2026.02.26/104218 \
    --dataset moses --num_samples 1000 --batch_size 500 --steps 1024 \
    --output samples/moses.txt

python tools/eval_metrics.py samples/moses.txt --train data/moses_train.txt --sample 1000
```

## 7. Prefix / suffix constrained generation

```bash
python sample_cond.py --model_path exp_local/moses/2026.02.26/104218 \
    --dataset moses --prefix "CC" --num_samples 1000 --wanti 1.0
```

The prefix is re-written into the sequence at every denoising step, so all samples
start with it; the default suffix is the end token `$`.

---

## 8. Property-targeted conditional generation (MCTS)

```bash
python -m condgen.run_mcts --variant puct --dataset moses \
    --model_path exp_local/moses/2026.02.26/104218 \
    --wanti 1.0 --simulations 500 --discovery_bs 100 --c_param 0.618 --seed 42
```

Each run writes `condgen_out/<dataset>_<variant>_logp<target>_<time>.{csv,json}`;
the CSV holds one row per simulation (`reward`, `avg_logp`, `mae`, `best_so_far`),
which is exactly what the figures need.

### 8.1 Variants (the ablations of the paper)

All variants are the same engine with four switches — `selection`, `prior`,
`dynamic_vocab`, `validity_filter`:

| `--variant`            | selection | prior      | vocabulary | validity filter | role |
|------------------------|-----------|------------|------------|-----------------|------|
| `puct`                 | PUCT      | SEDD prior | dynamic    | yes             | **proposed method** |
| `uct`                  | UCT       | uniform    | dynamic    | yes             | UCT + dynamic vocabulary |
| `uct_novalid`          | UCT       | uniform    | dynamic    | no              | removes the validity filter |
| `pure`                 | UCT       | uniform    | static     | —               | pure MCTS baseline |
| `puct_uniform`         | PUCT      | uniform    | dynamic    | yes             | removes the SEDD prior |
| `puct_novalid`         | PUCT      | SEDD prior | dynamic    | no              | prior only |
| `puct_uniform_novalid` | PUCT      | uniform    | dynamic    | no              | neither |

Any combination can also be set explicitly:

```bash
python -m condgen.run_mcts --variant puct --selection uct --prior uniform \
    --vocab static --validity_filter off ...
```

### 8.2 Classical baselines

```bash
python -m condgen.run_baselines --method greedy --dataset qm9 --model_path ... --wanti 1.0
python -m condgen.run_baselines --method beam   --dataset qm9 --model_path ... --beam_width 3
python -m condgen.run_baselines --method beam_mcts --dataset qm9 --model_path ... \
    --beam_width 5 --beam_depth 3 --simulations 500
```

### 8.3 Hyper-parameters used in the paper

| dataset | `max_steps` | `simulations` | `c_param` | `discovery_bs` | start tokens |
|---------|-------------|---------------|-----------|----------------|--------------|
| MOSES   | 46          | 500 (3000 for the full sweep) | 0.618 | 100 | `Br,C,Cl,F,N,O,[H],c` |
| QM9     | 17          | 1500 / 2000   | 1.414 / 0.618 | 200        | `C,N,O,F` |
| ZINC    | 47          | 1500          | 1.414     | 200            | `C,c,N,O` |

See [`docs/hyperparameters_zh.md`](docs/hyperparameters_zh.md) for the complete list
(model size, optimizer, sampling steps, …).

---

## 9. Figures

```bash
# four-line ablation (best-so-far reward)
python figures/plot_ablation.py \
  --run "SEDD-prior MCTS (ours)=condgen_out/moses_puct_*.csv" \
  --run "UCT + dynamic vocab=condgen_out/moses_uct_*.csv" \
  --run "UCT, no validity filter=condgen_out/moses_uct_novalid_*.csv" \
  --run "Pure MCTS=condgen_out/moses_pure_*.csv" \
  --output results/ablation.pdf

python figures/plot_speed.py --csv results/speed_sweep.csv
python figures/plot_loss.py --log exp_local/moses/2026.02.26/104218/logs
```

`figures/paper/` contains the camera-ready scripts (Nature-style multi-panel
figures, t-SNE, SA/RAscore panels) together with `figure_captions.txt` and
`source_data_summary.csv`. They expect the original plotting data layout described
in their headers.

`results/` keeps the search logs behind the published curves
(`results/logs/moses_*.txt`, `*.log`) and the 5-seed results table
(`results/moses_5seed_results_table.md`), so the figures can be regenerated without
re-running the search.

---

## 10. Speed benchmark

```bash
python tools/benchmark_speed.py --model_path exp_local/moses/2026.02.26/104218 \
    --dataset moses --batches 1,10,100,500,1000,2000 --steps 1024 \
    --output results/speed_sweep.csv
```

Reference numbers and the interpretation of the curve are in
[`docs/speed_benchmark_zh.md`](docs/speed_benchmark_zh.md).

---

## 11. Mapping from the original experiment scripts

| original file | in this repository |
|---|---|
| `MOSES_1_our.py` | `condgen/run_mcts.py --variant puct` |
| `MOSEScond_GGTMCTS.py` | `condgen/run_mcts.py --variant uct` |
| `MOSES_5_UCT无有效性检测.py` | `condgen/run_mcts.py --variant uct_novalid` |
| `MOSEScond_PureMCTS.py` | `condgen/run_mcts.py --variant pure` |
| `MOSES_2_减去频率.py` | `condgen/run_mcts.py --variant puct_uniform` |
| `MOSES_3_减去有效性检测.py` | `condgen/run_mcts.py --variant puct_novalid` |
| `MOSES_4_减去有效性检测和频率.py` | `condgen/run_mcts.py --variant puct_uniform_novalid` |
| `QM9cond_BeamSearch.py` | `condgen/run_baselines.py --method beam` |
| `QM9cond_greedy.py` | `condgen/run_baselines.py --method greedy` |
| `QM9cond_BeamMCTS.py` | `condgen/run_baselines.py --method beam_mcts` |
| `run_sample_MOSES.py` / `run_sample.py` | `sample.py --dataset moses/qm9` |
| `run_sample_cond_MOSES.py` | `sample_cond.py --dataset moses` |
| `eval_metrics.py` | `tools/eval_metrics.py` |
| `get_*_data.py`, `count_tokens.py` | `tools/prepare_data.py` |
| `benchmark_speed_sweep.py` | `tools/benchmark_speed.py` |

Dataset-specific constants (`total_len`, vocabulary, end token, default start tokens)
are centralised in `sedd_mol/vocab.py` instead of being copied into every script.

---

## 12. Citation

If you find this code useful, please cite our paper (bibtex to be added) and the
original SEDD paper:

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

## 13. Licence

MIT — see [LICENSE](LICENSE). The code in `sedd/` derives from
[SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) (MIT).

---

## 中文速览

* 训练：`python train.py --config-name config_moses`（QM9 / ZINC / MOSES 三套配置在 `configs/`）
* 无条件生成：`python sample.py --dataset moses --model_path <实验目录>`
* 前缀/后缀条件生成：`python sample_cond.py --prefix "CC" ...`
* 性质定向条件生成（论文主方法 = SEDD 先验 PUCT）：`python -m condgen.run_mcts --variant puct ...`
  其余消融用 `--variant uct / uct_novalid / pure / puct_uniform / puct_novalid / puct_uniform_novalid`
* 原脚本与本仓库的对应关系见第 11 节；数据集词表统一放在 `sedd_mol/vocab.py`。
