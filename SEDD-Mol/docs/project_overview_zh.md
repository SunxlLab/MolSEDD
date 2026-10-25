# Score Entropy Discrete Diffusion 项目文件总览

> 本文档总结了项目中所有文件的功能和作用，供 agent 编写功能模块时快速定位参考。

---

## 一、项目概述

本项目实现基于 **Score Entropy Discrete Diffusion (SEDD)** 的分子生成模型。核心思想是将离散扩散模型应用于 SMILES 分子字符串的生成，支持无条件生成和属性条件生成（通过 MCTS/BeamSearch 等方法控制 LogP、QED 等分子属性）。

涉及三个数据集：**QM9**（小分子）、**ZINC**（类药物分子）、**MOSES**（基准分子集）。

---

## 二、核心模型模块 (`model/`)

| 文件 | 作用 | 关键内容 |
|------|------|----------|
| `model/__init__.py` | 包入口，导出 `SEDD` 类 | `from .transformer import SEDD` |
| `model/transformer.py` | **核心模型定义** | `SEDD` 类（继承 `nn.Module` + `PyTorchModelHubMixin`）：DiT架构的离散扩散Transformer。包含 `DDiTBlock`（带adaLN的DiT Block）、`TimestepEmbedder`（正弦时间步嵌入）、`EmbeddingLayer`（词嵌入）、`DDitFinalLayer`（输出层）。使用 Flash Attention 加速。 |
| `model/ema.py` | **指数移动平均 (EMA)** | `ExponentialMovingAverage` 类：维护模型参数的EMA影子参数，支持 `update`/`copy_to`/`store`/`restore` 操作，用于推理时的参数平滑。 |
| `model/rotary.py` | **旋转位置编码 (RoPE)** | `Rotary` 类：生成 cos/sin 缓存；`apply_rotary_pos_emb`：优先使用 flash_attn 的融合实现。 |
| `model/fused_add_dropout_scale.py` | **融合算子** | `bias_dropout_add_scale`：残差连接+dropout+缩放；`modulate`：adaLN条件调制。提供训练/推理两种融合版本（`_fused_train`/`_fused_inference`）。 |
| `model/utils.py` | **模型工具函数** | `get_model_fn`：包装训练/评估模式的模型前向；`get_score_fn`：获取 score function（训练返回 log-score，采样返回 exp(score) 即真实 score）。 |

---

## 三、扩散过程核心模块

### 3.1 `graph_lib.py` — 离散状态转移图
定义离散扩散中的状态转移矩阵（rate matrix）。核心类：

| 类/函数 | 作用 |
|---------|------|
| `Graph` (抽象类) | 定义接口：`rate`（前向速率列）、`transp_rate`（转置速率行）、`transition`（转移矩阵）、`reverse_rate`（反向速率 = score * transp_rate）、`staggered_score`（交错评分）、`score_entropy`（评分熵损失）、`sample_limit`（极限分布采样） |
| `Uniform` | **均匀扩散图**：每个token均匀转移到其他所有token。用于 QM9 数据集。 |
| `Absorbing` | **吸收态扩散图**：token转移到 [MASK] 吸收态（`dim-1`）。用于 MOSES/ZINC 数据集。 |
| `get_graph()` | 根据配置选择图类型（`uniform` 或 `absorb`） |

### 3.2 `noise_lib.py` — 噪声调度
定义扩散过程中的噪声时间调度（noise schedule）。核心类：

| 类/函数 | 作用 |
|---------|------|
| `Noise` (抽象类) | 定义接口：`total_noise(t)` 返回累积噪声 σ(t)，`rate_noise(t)` 返回噪声变化率 dσ/dt |
| `GeometricNoise` | **几何噪声调度**：σ(t) = σ_min^(1-t) * σ_max^t。适用于 Uniform 扩散图。 |
| `LogLinearNoise` | **对数线性噪声调度**：σ(t) = -log(1 - (1-ε)·t)。适用于 Absorbing 扩散图。 |
| `get_noise()` | 根据配置选择噪声类型 |

### 3.3 `losses.py` — 损失函数与优化
| 函数 | 作用 |
|------|------|
| `get_loss_fn()` | 构建损失函数：对 batch 采样时间 t，加噪得到 perturbed_batch，计算 score entropy loss |
| `get_optimizer()` | 创建优化器（Adam/AdamW） |
| `optimization_manager()` | 优化步骤管理：warmup、gradient clipping、AMP scaler |
| `get_step_fn()` | 构建单步训练/评估函数：支持梯度累积，EMA更新 |

### 3.4 `sampling.py` — 采样逻辑
| 类/函数 | 作用 |
|---------|------|
| `Predictor` (抽象类) | 预测器基类 |
| `EulerPredictor` | **欧拉预测器**：用反向速率做单步采样 |
| `AnalyticPredictor` | **解析预测器**：使用 staggered_score + transp_transition 计算精确转移概率 |
| `Denoiser` | 去噪器：在最后一步将吸收态去噪 |
| `get_pc_sampler()` | **Predictor-Corrector 采样器**：从极限分布开始，逐步迭代去噪生成样本 |
| `get_sampling_fn()` | 根据配置构建采样函数 |

### 3.5 `sampling_train.py` — 带均值方差控制的采样
与 `sampling.py` 类似，但增加了 `sample_discrete_tokens_exact()` 函数：生成具有指定均值(param_a)和方差(param_b)的离散初始token分布，替代默认的极限分布。

### 3.6 `catsample.py` — 分类采样工具
提供从分类概率分布中采样的工具函数：`sample_categorical()`（hard Gumbel方法）。

---

## 四、训练与推理入口

| 文件 | 作用 | 关键内容 |
|------|------|----------|
| `train.py` | **训练主入口** | 使用 Hydra 加载配置，通过 `mp.spawn` 启动多GPU分布式训练。训练完成后生成 `config.json`。 |
| `run_train.py` | **训练执行逻辑** | `run_multiprocess()`：初始化分布式环境，构建 graph/noise/model/optimizer/EMA，执行训练循环（包含定期保存checkpoint、日志记录、采样评估）。 |
| `run_sample.py` | **无条件采样脚本** | 加载模型，执行无条件采样，将生成的 token IDs 转换为 SMILES 字符串，计算有效性，保存结果到 `Test_text/` |
| `run_sample_cond.py` | **条件采样脚本（QM9）** | 基于前缀/后缀约束的条件生成。使用 `proj_fun` 将指定位置的 token 固定为输入前缀/后缀。 |

---

## 五、数据集处理模块

### 5.1 主数据加载 (`data.py`)
- 从 QM9 数据集加载 SMILES，分词后映射为 token IDs（词表21个token）
- 过滤长度在 [6, 22] 的序列，padding 到 22 位
- 构建无限循环的 DataLoader（`cycle_loader`）

### 5.2 数据处理变体 (`data处理项关/`)

| 文件 | 作用 | 差异点 |
|------|------|--------|
| `data-短文本.py` | QM9短文本处理 | 长度过滤 [14, 18]，padding到18位 |
| `data-长文本.py` | QM9长文本处理（无padding） | 使用 `group_texts` 拼接变长序列为固定 block_size 块 |
| `data-原始作者方法.py` | 原始论文数据处理方式 | 与 `data-长文本.py` 类似，保持原始方法 |
| `data-新数据集短文本.py` | ZINC短文本处理 | 词表63个token，长度过滤 [29, 48]，padding到48位 |
| `data-新数据集长文本.py` | ZINC长文本处理 | 与短文本同词表，使用 `group_texts` 拼接方式 |
| `data-MOSES.py` | MOSES数据处理 | 词表25个token，长度过滤 [29, 44]，padding到44位 |
| `data-类似长度长文本.py` | 类似长度长文本 | 变体版本 |
| `data-长文本padding方法.py` | 长文本padding方法 | 变体版本 |
| `extract_filtered_moses.py` | MOSES数据提取 | 从已过滤数据中提取 |

### 5.3 数据获取脚本

| 文件 | 作用 |
|------|------|
| `get_moses_data.py` | 读取 `MOSES.txt`，按长度过滤后保存到 `todoimage/trainMOSES.txt` |
| `get_qm9_data.py` | 获取 QM9 数据集 |
| `get_zinc_data.py` | 读取 `250k_rndm_zinc_drugs_clean.smi`，按ZINC词表分词+长度过滤，保存到 `todoimage/trainZINC.txt` |
| `check_moses_len.py` | 检查 MOSES 数据长度分布 |
| `clean_train_data.py` | 清洗训练数据：移除 `,train` 后缀 |
| `count_tokens.py` | 统计 SMILES 数据集的词汇表、每个位置的词表、最大长度（多进程并行） |
| `count_zinc_tokens.py` | 专门统计 ZINC 数据集的 token |

---

## 六、条件生成与搜索 (MCTS/BeamSearch)

### 6.1 条件 MCTS 搜索

| 文件 | 作用 | 数据集 |
|------|------|--------|
| `QM9cond_MCTS.py` | QM9条件MCTS | QM9（QM9词表，LogP条件） |
| `QM9cond_GGMCTS.py` | QM9条件GGMCTS | QM9（带梯度引导的MCTS） |
| `QM9cond_GGTMCTS.py` | QM9条件GGTMCTS | QM9（GGT=Gradient Guided Tree） |
| `QM9cond_BeamMCTS.py` | QM9条件BeamMCTS | QM9（Beam Search + MCTS） |
| `QM9cond_BeamSearch.py` | QM9条件BeamSearch | QM9（纯Beam Search） |
| `QM9cond_greedy.py` | QM9条件贪心搜索 | QM9（贪心解码） |
| `MOSEScond_MCTS.py` | MOSES条件MCTS | MOSES（MOSES词表，LogP条件） |
| `MOSEScond_GGTMCTS.py` | MOSES条件GGTMCTS | MOSES（GGT MCTS） |
| `MOSEScond_GGTMCTScopy.py` | MOSES条件GGTMCTS副本 | MOSES（GGT MCTS变体/备份） |
| `ZINCcond_MCTS.py` | ZINC条件MCTS | ZINC（ZINC词表，LogP条件） |

**核心逻辑**：这些脚本共享类似的 `MolecularGenerator` 类，流程为：
1. 加载预训练SEDD模型
2. 对给定前缀（如 "C"），使用扩散模型生成候选分子
3. 通过 MCTS 树搜索选择最大化目标属性（如 LogP）的 token 序列
4. 使用 RDKit 验证分子有效性并计算属性值

### 6.2 run_sample 系列脚本

| 文件 | 作用 |
|------|------|
| `run_sample.py` | 通用无条件采样 |
| `run_sample-短文本.py` | 短文本（18位）无条件采样 |
| `run_sample-长文本.py` | 长文本无条件采样 |
| `run_sample-删除头尾字符串版本.py` | 去掉首尾约束的采样 |
| `run_sample-新数据集短文本.py` | ZINC短文本采样 |
| `run_sample-新数据集长文本.py` | ZINC长文本采样 |
| `run_sample_MOSES.py` | MOSES数据集采样 |
| `run_sample_cond.py` | 条件采样（QM9，前缀约束） |
| `run_sample_cond-长文本.py` | 条件采样长文本版本 |
| `run_sample_cond1.py` | 条件采样变体v1 |
| `run_sample_cond_MOSES.py` | MOSES条件采样 |
| `run_sample_cond_zinc.py` | ZINC条件采样 |
| `run_sample_cond_zinc1.py` | ZINC条件采样v1 |
| `run_sample_cond_zinc2.py` | ZINC条件采样v2 |

---

## 七、损失函数变体 (`loss处理相关/`)

| 文件 | 作用 | 差异 |
|------|------|--------|
| `losses-原始.py` | 原始损失函数 | 基线版本 |
| `losses-添加准确率影响损失版本.py` | 加入准确率影响的损失 | 可能将分子有效性纳入损失 |
| `losses-短文本最新.py` | 短文本最新损失 | 适配短文本（18位padding）的损失 |
| `losses-长文本padding.py` | 长文本padding损失 | 适配长文本padding的损失 |
| `losses-长文本最新.py` | 长文本最新损失 | 长文本最新版本 |

---

## 八、工具与辅助模块

| 文件 | 作用 |
|------|------|
| `utils.py` | 通用工具：`load_hydra_config_from_run`（加载Hydra配置）、`makedirs`、`get_logger`、`restore_checkpoint`/`save_checkpoint`（断点续训） |
| `load_model.py` | **模型加载器**：支持从 HuggingFace Hub 加载（`load_model_hf`）或本地 checkpoint 加载（`load_model_local`），自动恢复 EMA 参数 |
| `c_logp.py` | LogP 计算工具：使用 RDKit 的 `Crippen.MolLogP` 计算分子的脂水分配系数。也包含 SMILES 有效性检查、RDKit 警告控制。 |
| `plot_loss.py` | 训练损失可视化：从训练日志解析 training_loss 和 evaluation_loss，绘制损失曲线并保存。 |
| `setup.py` | Python 包安装脚本 |

---

## 九、配置文件 (`configs/`)

### 9.1 主配置文件
| 文件 | 作用 |
|------|------|
| `config.yaml` | 默认配置（`configs/config.yaml`） |
| `config copy.yaml` | 配置副本/备份 |

### 9.2 场景配置
| 文件 | 适配场景 |
|------|---------|
| `config-短文本.yaml` | 短文本（18位）条件生成 |
| `config-长文本.yaml` | 长文本条件生成 |
| `config-短文本-新数据集.yaml` | ZINC短文本条件生成 |
| `config-长文本-新数据集.yaml` | ZINC长文本条件生成 |
| `config-un-短文本.yaml` | 短文本无条件（unconditional） |
| `config-un-长文本.yaml` | 长文本无条件 |
| `config-un-短文本-新数据集.yaml` | ZINC短文本无条件 |
| `config-un-长文本-新数据集.yaml` | ZINC长文本无条件 |

### 9.3 模型配置 (`configs/model/`)
| 文件 | 作用 |
|------|------|
| `small.yaml` | 小模型配置（条件模型，吸收态图） |
| `small_un.yaml` | 小模型无条件配置（均匀扩散图） |
| `medium.yaml` | 中等模型条件配置 |
| `medium_un.yaml` | 中等模型无条件配置 |

**配置项关键字段**：
- `graph.type`: `"uniform"` 或 `"absorb"`
- `noise.type`: `"geometric"` 或 `"loglinear"`
- `model`: hidden_size, n_heads, n_blocks, cond_dim, dropout, length, scale_by_sigma
- `training`: batch_size, accum, n_iters, lr, ema, warmup, grad_clip, log_freq, eval_freq, snapshot_freq
- `sampling`: predictor (`"euler"`/`"analytic"`), steps, noise_removal
- `tokens`: 词表大小

---

## 十、第三方库与子项目

### 10.1 `qm9pack/`
QM9 数据处理包。核心文件：
- `qm9pack_getdata.py`：数据获取接口
- `qm9pack_utility.py`：工具函数
- `data/qm9_part[1-3].csv`：QM9 数据集分片（共 ~214MB）

### 10.2 `qm9pack-main/`
QM9Pack 教程/练习项目。包含12个练习脚本（`exercise_A01.py` ~ `exercise_C04.py`）和教程PDF。

### 10.3 `todoimage/RAscore/`
完整的 **RAscore（合成可及性评分）** 子项目（64个文件）：
- DNN 和 XGBoost 模型权重
- RAscore 计算与密度图绘制
- 用于评估生成分子的合成可及性

---

## 十一、可视化与分析 (`todoimage/`)

### 11.1 核心绘图脚本
| 文件 | 作用 |
|------|------|
| `ComputedMetrics.py` | 计算分子属性指标（LogP, QED, TPSA, SA等） |
| `cond_logp1234.py` / `condlogp.py` | 条件LogP分析 |
| `draw_smiles.py` | 绘制SMILES分子结构图 |
| `faceted_tsne.py` | 分面 t-SNE 可视化 |
| `faceted_tsne_zinc_moses.py` | ZINC/MOSES联合t-SNE |
| `plot_4_distributions.py` | 四个属性分布图 |
| `select_top20_by_logp_and_draw.py` | 选Top20 LogP分子并绘图 |
| `single_dataset_metrics.py` | 单数据集指标计算 |
| `toimage.py` | 数据转图片工具 |

### 11.2 图片输出目录
| 目录 | 内容 |
|------|------|
| `todoimage/image/` | 训练集 vs 生成集对比图（属性分布） |
| `todoimage/goodimage/` | 论文用高质量图（PDF/PNG） |
| `todoimage/tsneimage/` | t-SNE 降维可视化图 |
| `todoimage/smilesimg/` | Top分子 SMILES 结构图（含各条件组子目录） |
| `todoimage/condtsne/` | 条件生成的 t-SNE 数据 |
| `todoimage/zhaiyao/` | 论文摘要/表格图 |
| `todoimage/LZPneed/` | SA/RA 密度曲线图 |
| `todoimage/tokencount/` | 各数据集 token 统计文本 |

### 11.3 根目录图片 (`image/`)
LogP、QED、TPSA 属性在 QM9/ZINC/MOSES 三数据集上的分布图（含 -10k 子集版本）。

---

## 十二、实验日志与输出

### 12.1 条件生成日志
| 目录 | 内容 |
|------|------|
| `condition_log_MOSES/` | MOSES 各目标 LogP 值的 GGTMCTS/MCTS 生成日志 |
| `condition_log_QM9/` | QM9 各方法（MCTS/GGMCTS/GGTMCTS/BeamMCTS/greedy）在 LogP 条件 -1/0/1 下的生成日志 |
| `condition_log_ZINC/` | ZINC 条件生成日志 |

### 12.2 采样输出 (`Test_text/`)
各时间戳的采样输出文件（SMILES 字符串及 token ID 数组）。

### 12.3 训练日志
- `logs_ckpt_sweep/summary.csv`：checkpoint 扫描结果
- `loss_plots/`：损失曲线图

### 12.4 大数据文件（根目录）
| 文件 | 说明 |
|------|------|
| `250k_rndm_zinc_drugs_clean.smi` (10.78 MB) | ZINC 250K 药物分子SMILES |
| `All_all_mapped_token_ids_with_end.txt` (4.93 MB) | 全部映射后的token IDs |
| `all_mapped_token_ids_with_end.txt` (4.89 MB) | 映射后的token IDs |
| `MOSES.txt` (6.19 MB) | MOSES 数据集SMILES |
| `TRMOSES.txt` (55.7 MB) | 训练用MOSES数据（大文件） |

---

## 十三、测试脚本

| 文件 | 作用 |
|------|------|
| `test-long.py` | 长文本生成测试 |
| `test-padding.py` | Padding 逻辑测试 |

---

## 十四、其他

| 文件 | 作用 |
|------|------|
| `environment.yml` | Conda 环境依赖配置 |
| `LICENSE` | 开源许可证 |
| `run_all_checkpoints.sh` | Shell脚本：批量运行所有 checkpoint 的采样 |
| `Untitled.ipynb` | Jupyter Notebook（实验记录） |
| `losses copy.py` | losses.py 的备份副本 |

---

## 十五、关键数据流总结

```
数据集(SMILES) 
    → [data.py / data处理项关/] 分词+ID映射+padding 
    → DataLoader 
    → [train.py + run_train.py] 
        → SEDD模型 (model/transformer.py)
        → Graph (graph_lib.py, Uniform/Absorbing)
        → Noise (noise_lib.py, Geometric/LogLinear)
        → Loss (losses.py, score entropy)
        → 训练循环 + EMA + Checkpoint
    → 模型保存 
    → [run_sample.py / run_sample_cond.py / MOSEScond_MCTS.py等]
        → load_model.py 加载
        → sampling.py 采样/去噪
        → token IDs → SMILES 字符串
        → RDKit 验证 + 属性计算
    → 输出/评估
```

---

> 最后更新：2026-07-25
