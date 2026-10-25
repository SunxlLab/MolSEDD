# 论文超参数汇总

只收录论文中需要展示的超参数。
判定原则：超参数必须是**数值型、可调节、更换数值会影响训练/生成结果**的量。
架构类型、布尔开关、随机种子、数据集名称等不属于超参数，不收录。

"展示位置"列：正文（Table / 方法段落）/ 附录。

---

## 1. 模型结构超参数

来源：`configs/model/medium_un.yaml`

| 参数 | 值 | 展示位置 |
|---|---|---|
| 隐藏维度 `hidden_size` | 1024 | 正文（Table） |
| Transformer 块数 `n_blocks` | 24 | 正文（Table） |
| 注意力头数 `n_heads` | 16 | 正文（Table） |
| 条件嵌入维度 `cond_dim` | 128 | 附录 |
| 序列长度 `length` | 22 | 正文（Table） |
| Dropout | 0.1 | 正文（Table） |

> 注：三个模型（short/long/un 等）的 `length` 不同，正文统一报告本表值即可，差异在附录说明。

---

## 2. 训练流程超参数

来源：`configs/config.yaml → training`

| 参数 | 值 | 展示位置 |
|---|---|---|
| 总训练步数 `n_iters` | 10000 | 正文（方法段） |
| Batch size | 1024 | 正文（方法段） |
| EMA 衰减 `ema` | 0.9999 | 附录 |
| 词表大小 `tokens` | 22 | 正文（方法段） |

---

## 3. 优化器超参数

来源：`configs/config.yaml → optim`

| 参数 | 值 | 展示位置 |
|---|---|---|
| 优化器类型 | AdamW | 正文（方法段） |
| 学习率 `lr` | 2e-4 | 正文（方法段） |
| Weight decay | 0.01 | 正文（Table） |
| Warmup 步数 | 2000 | 正文（Table） |
| Beta1 / Beta2 / eps | 0.9 / 0.999 / 1e-8 | 附录 |
| 梯度裁剪 `grad_clip` | 1.0 | 附录 |

---

## 4. 离散扩散设置

来源：`configs/config.yaml → graph`

| 参数 | 值 | 展示位置 |
|---|---|---|
| 前向图类型 `graph.type` | uniform | 正文（方法段） |

---

## 5. MCTS 条件生成超参数（论文方法核心）

来源：`MOSEScond_SEDDweighted_MCTS.py`（方案 A / SEDD-weighted）、`MOSEScond_GGTMCTS.py`（方案 B / Uniform）、`MOSEScond_PureMCTS.py`（全词表 UCT / Full-vocabulary）。

| 参数 | 值 | 说明 | 展示位置 |
|---|---|---|---|
| MCTS 模拟次数 `--simulations` | 3000 | 每次运行的总搜索轮数 | 正文（Table） |
| 探索常数 `c_param` | 0.618（PUCT）/ 1.414（UCT） | SEDD-weighted 用 0.618，GGTMCTS/纯 MCTS 用 1.414 | 正文（Table） |
| 子节点发现 batch `--discovery_bs` | 200 | 每次扩展生成的分子数 | 正文（Table） |
| 扩散采样步数（生成时） | 1024 | 每个前缀补全的采样步数 | 正文（Table） |
| 最大前缀长度 `max_steps` | 46 | MOSES total_len=47，留 1 位给结束符 `$` | 附录 |

---

## 6. 论文呈现建议

- **正文 Table 放两处**：① MCTS 参数表（第 5 节，创新点）；② 模型规模一行（hidden_size / n_blocks / n_heads / length / dropout / 参数量）。
- **训练配置在方法段落 1-2 句话带过**：如 "The model was trained with AdamW (lr=2e-4, weight decay 0.01, 2000 warmup steps) for 10k steps with batch size 1024."，无需单独成表。
- **EMA、beta、梯度裁剪、cond_dim、max_steps 等放附录**，保证可复现。
- 随机种子、数据集名称、架构类型、布尔开关等不列入超参数表。
