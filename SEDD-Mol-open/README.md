# MolSEDD

基于 **score-entropy discrete diffusion (SEDD)** 的分子生成与性质导向搜索。代码分五块：

| 模块 | 目录 | 做什么 |
|---|---|---|
| ① 模型训练 | `train.py` + `configs/` | 在 QM9 / ZINC-250k / MOSES 上训练 SEDD |
| ② 无条件生成 | `generate/` | 加载训练好的 checkpoint，直接采样 SMILES |
| ③ 条件生成 | `conditional/` | 给定 SMILES 前缀，让模型补全剩下的部分 |
| ④ MCTS 性质定向 | `mcts/` | 用 SEDD 自己的下一 token 分布做 PUCT 先验，朝目标 logP 搜索 |
| ⑤ 消融实验 | `ablation/` | 论文图 5 的四条曲线 |

`sedd/` 是离散扩散核心（图 / 噪声 / 采样器 / 损失 / 模型），`sedd_mol/` 是分子层
（词表 / 数据集 / 前缀投影生成器 / logP 奖励），这两块是上面五个模块共用的基础。

---

## 1. 方法

分子是 SMILES 字符串，也就是一段离散 token 序列。SEDD 直接学习数据分布在各噪声
水平上的比值，用 score-entropy 目标训练，采样是 τ-leaping 式的 analytic 反向过程
（`sedd/sampling.py`）。

**性质导向生成**：模型完全冻结，用 MCTS 逐 token 生长 SMILES 前缀。每个节点把当前
前缀钉住采样一批分子，一次调用同时拿到两样东西：

* reward `R = exp(-MAE)` —— 这批分子的 Crippen logP 与目标值的偏差；
* 下一 token 的经验分布 `P(s,a)` —— 只从这批分子实际产生过的 token 里选动作，
  并用模型自己的频率作为 PUCT 先验。

选择规则：

```
Q(s,a) + c · P(s,a) · √N(s) / (1 + n(s,a))        (PUCT，本文方法)
Q(s,a) + c · √(2 ln N(s) / n(s,a))                (UCT，基线)
```

推导见 [`docs/mcts_formulas.md`](docs/mcts_formulas.md)。

搜索结束后取奖励最高的前缀，用 `conditional/` 里的脚本以该前缀为条件生成 1000 个
分子，作为最终评估样本。

代码基于 Lou et al. 的 SEDD：

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

---

## 2. 目录结构

```
.
├── train.py                     ① 训练入口（Hydra）
├── configs/
│   ├── config.yaml              QM9 + uniform（默认）
│   ├── config_zinc.yaml         ZINC-250k
│   ├── config_moses.yaml        MOSES
│   ├── config_qm9_absorb.yaml   QM9 + absorbing（论文未用，仅对照）
│   └── model/                   DDiT backbone 尺寸
│
├── generate/                    ② 无条件生成
│   ├── sample_qm9.py
│   ├── sample_zinc.py
│   └── sample_moses.py
│
├── conditional/                 ③ 条件生成（前缀投影）
│   ├── cond_qm9.py
│   ├── cond_zinc.py
│   └── cond_moses.py
│
├── mcts/                        ④ MCTS 性质定向
│   ├── search.py                搜索引擎（选择 / 扩展 / 回传 / reward）
│   ├── run_qm9.py               论文主方法，每个数据集一份
│   ├── run_zinc.py
│   └── run_moses.py
│
├── ablation/                    ⑤ 消融实验
│   ├── run_ablation.py          论文图 5 的四条曲线
│   └── run_ablation.sh
│
├── sedd/                        离散扩散核心
│   ├── graph_lib.py               uniform / absorbing 前向过程
│   ├── noise_lib.py               几何 / log-linear 噪声表
│   ├── sampling.py                预测器、去噪器、采样器（含 proj_fun 投影钩子）
│   ├── catsample.py               类别采样
│   ├── losses.py                  score-entropy 损失、优化器、训练步
│   ├── load_model.py              从实验目录 / HF hub 加载模型
│   ├── run_train.py               DDP 训练 worker（由 train.py spawn）
│   ├── utils.py                   日志与 checkpoint
│   └── model/                     Transformer backbone（DDiT）+ EMA
│
├── sedd_mol/                    分子层
│   ├── vocab.py                   三个数据集的冻结词表与长度规格
│   ├── dataset.py                 SMILES → token id → DataLoader
│   ├── generator.py               前缀投影采样器 MolecularGenerator
│   └── metrics.py                 Crippen logP 与 MCTS 的目标性质奖励
│
├── tools/prepare_data.py        构建训练语料
│
└── docs/                        MCTS 搜索的公式与推导
```

每个生成脚本都是独立可读的：数据集长度、词表、默认超参都写在该文件顶部。

---

## 3. 数据集规格

三个数据集的词表和序列长度各不相同，具体数值固化在 `sedd_mol/vocab.py` 和各
`configs/config_*.yaml` 里。生成脚本顶部也各自写了一份自己数据集的配置，
改的时候三处要一起改。

| 数据集 | 词表构成 | 序列长度 | 训练长度过滤 |
|---|---|---|---|
| QM9 | 20 SMILES token + pad + end = 22 | 22 | 6 – 22 |
| ZINC-250k | 62 SMILES token + pad + end = 64 | 48 | 29 – 48 |
| MOSES | 26 SMILES token + pad + end = 28 | 47 | 24 – 47 |

---

## 4. 安装

```bash
conda env create -f environment.yml      # 创建 "sedd" 环境（CUDA 11.8）
conda activate sedd
```

或者用已有的 PyTorch：

```bash
pip install -r requirements.txt
```

`flash-attn` 是**可选**的：导入失败时自动回退到
`torch.nn.functional.scaled_dot_product_attention`（`sedd/model/transformer.py`）。

---

## 5. 数据准备

token 化的语料不入库（QM9 一个就 215 MB），需要自己下载原始数据集并转换：

```bash
# QM9 — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
python tools/prepare_data.py --dataset qm9  --input qm9.csv  --output data/qm9_train.txt --stats

# ZINC 250k — https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean.smi
python tools/prepare_data.py --dataset zinc --input 250k_rndm_zinc_drugs_clean.smi \
    --output data/zinc_train.txt --stats

# MOSES — https://github.com/molecularsets/moses （train split）
python tools/prepare_data.py --dataset moses --input train.txt --output data/moses_train.txt --stats
```

词表在 `sedd_mol/vocab.py` 里是**硬编码**的：发布出来的 checkpoint 就是按这个
index 顺序训练的，从别的语料重新生成词表会让所有 embedding 错位。

---

## 6. 训练

```bash
python train.py                                   # QM9 + uniform（默认）
python train.py --config-name config_zinc          # ZINC-250k
python train.py --config-name config_moses         # MOSES
python train.py --config-name config_moses ngpus=4 training.batch_size=1024
```

Hydra 会把每次运行写到 `exp_local/<dataset>/<date>/<time>/`：

```
exp_local/moses/2026.01.01/120000/
├── .hydra/config.yaml                 # load_model 需要
├── checkpoints/checkpoint_*.pth
├── checkpoints-meta/checkpoint.pth    # 断点续训 / load_model
├── samples/
├── logs
└── config.json
```

三套配置的超参数都在 `configs/config_*.yaml` 里，论文附录提供了完整列表。

---

## 7. 无条件生成

三个脚本对应三个数据集，长度和词表各自写死：

```bash
python generate/sample_qm9.py   --model_path exp_local/qm9/2026.01.01/120000 \
    --num_samples 10000 --steps 1024

python generate/sample_zinc.py  --model_path exp_local/zinc/2026.01.01/120000 \
    --num_samples 10000 --steps 1024

python generate/sample_moses.py --model_path exp_local/moses/2026.01.01/120000 \
    --num_samples 10000 --steps 1024
```

输出写到 `samples/<dataset>_<时间戳>.txt`，一行一个 SMILES。
`--num_samples` 默认就是论文口径的 **10000**。

---

## 8. 条件生成（前缀投影）

```bash
# QM9：官方 checkpoint 用过的示例前缀
python conditional/cond_qm9.py --model_path exp_local/qm9/2026.01.01/120000 \
    --prefix "FOCC1=C" --wanti 1.0 --num_samples 1000

# MOSES
python conditional/cond_moses.py --model_path exp_local/moses/2026.01.01/120000 \
    --prefix "CC" --wanti 1.0 --num_samples 1000

# ZINC
python conditional/cond_zinc.py --model_path exp_local/zinc/2026.01.01/120000 \
    --prefix "CC" --wanti 1.0 --num_samples 1000
```

原理就是"无条件生成 + 前缀"：每一步去噪后把前缀和结束符写回序列，模型只能补中间
空出来的位置。不需要额外参数，也不需要微调。脚本会打印前缀保留率；给了 `--wanti`
就额外报告这批分子的 avg logP / MAE / reward。

论文里这一步用于"取 MCTS 搜到的最优前缀，让 SEDD 补全出 1000 个分子"再做性质评估
（表 4），所以默认 1000 个。

---

## 9. MCTS 性质定向

```bash
python mcts/run_moses.py --model_path exp_local/moses/2026.01.01/120000 --wanti 1.0

python mcts/run_qm9.py   --model_path exp_local/qm9/2026.01.01/120000 --wanti 1.0
python mcts/run_zinc.py  --model_path exp_local/zinc/2026.01.01/120000 --wanti 1.0
```

每次运行会在 `mcts_out/` 下写一对文件：

* `<dataset>_puct_logp<target>_<time>.csv` — 每个 simulation 一行
  （`reward`、`avg_logp`、`mae`、`best_so_far`），画收敛曲线就用它；
* 同名 `.json` — 参数、耗时和三条最优路径的评估结果。

各数据集的默认搜索超参写在各脚本的 `DATASET_CONFIG` 里，都可以用命令行参数覆盖；
论文报告的是 MOSES 上的结果。

---

## 10. 消融实验

论文图 5 的四条曲线，都是同一个引擎的四个开关组合：

| `--variants` | selection | prior | 动作空间 | 有效性过滤 | 作用 |
|---|---|---|---|---|---|
| `puct` | PUCT | SEDD 先验 | 动态 | 有 | **MolSEDD（本文方法）** |
| `uct` | UCT | 均匀 | 动态 | 有 | 换成 UCT |
| `uct_novalid` | UCT | 均匀 | 动态 | 无 | 去掉有效性过滤 |
| `pure` | UCT | 均匀 | 静态全词表 | — | 纯 MCTS |

```bash
# 四条线全跑
python ablation/run_ablation.py --dataset moses --model_path exp_local/moses/2026.01.01/120000

# 只跑其中一条
python ablation/run_ablation.py --dataset moses --model_path ... --variants puct

# 或者用 shell 包装（每个变体一份 log）
MODEL_PATH=exp_local/moses/2026.01.01/120000 ./ablation/run_ablation.sh
```

搜索超参复用 `mcts/run_{qm9,zinc,moses}.py` 的 `DATASET_CONFIG`，只有每节点补全数
单独设为 100（`ABLATION_DISCOVERY_BS`），与论文 4.5 的消融设置一致。

---

## 11. 引用与许可

如果这份代码对你有用，请引用我们的论文（bibtex 待补）以及原始 SEDD：

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

MIT — 见 [LICENSE](LICENSE)。`sedd/` 目录的代码来自
[SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)（MIT）。
