# MCTS 搜索：奖励、选择规则与候选动作

本文记录 `mcts/search.py` 里用到的全部公式与判定规则。

## 符号

| 符号 | 含义 | 代码变量 |
|------|------|----------|
| $V(a)$ | 动作 $a$ 的累积奖励 | `child.value` |
| $N(a)$ | 动作 $a$ 被访问的次数 | `child.visits` |
| $Q(a)$ | $V(a)/N(a)$，动作的历史平均奖励 | `child.value / child.visits` |
| $N(s)$ | 父节点的总访问次数 | `self.visits` |
| $c$ | 探索常数 | `c_param` |
| $P(s,a)$ | SEDD 补全给出的动作先验 | `child.prior` |

---

## 1. 奖励

三种配置共用同一个奖励，基于生成批次相对目标 LogP 的平均绝对偏差：

$$R(s) = e^{-\operatorname{MAE}(s)}, \qquad
\operatorname{MAE}(s) = \frac{1}{|\mathcal{C}(s)|}\sum_{\mathbf{x}\in\mathcal{C}(s)}\left|\operatorname{LogP}(\mathbf{x}) - y^*\right|$$

- $y^*$：目标 LogP，通过 `--wanti` 指定；
- $\mathcal{C}(s)$：在前缀 $s$ 被钉住的条件下，SEDD 采样出的一批分子；
- $R \in (0, 1]$，批次越接近目标，奖励越接近 1。

**有效性门槛**：若该批次中 RDKit 能解析的比例不超过 `min_validity`（默认 0.5），
整批视为失败，$R = 0$，该节点标记为死亡（`is_dead`），不再参与后续选择。
这与论文 3.3 的"有效率低于 50% 的前缀被剪除"对应。

注意奖励是**整批的平均表现**，不是批次里单个最好分子的表现——这样能引导搜索走向
"稳定生成接近目标分子的区域"，而不是碰运气。

---

## 2. 三种配置的选择规则

### 2.1 纯 MCTS —— UCT + 静态词表

候选动作固定为整个 SMILES 词表，所有节点共享，SEDD 只用来评估前缀好坏。

$$U_{\mathrm{UCT}}(s,a) = Q(a) + c\sqrt{\frac{2\ln N(s)}{N(a)}}$$

### 2.2 MCTS + 动态词表 —— UCT

候选动作改为"SEDD 补全中实际出现过的下一个 token"，选择仍用上面的 UCT。
SEDD 此时参与候选构建与评估，但不参与动作排序。

### 2.3 MolSEDD —— PUCT + SEDD 先验

候选动作同上，另外把每个动作在补全中出现的经验频率作为先验：

$$C(s,a) = \sum_{\mathbf{x}\in\mathcal{C}(s)} \mathbb{I}\left[x_{|s|+1} = a\right],
\qquad
P(s,a) = \frac{C(s,a)}{\sum_{a'\in\mathcal{A}(s)} C(s,a')}$$

$$U_{\mathrm{PUCT}}(s,a) = Q(a) + c\,P(s,a)\,\frac{\sqrt{N(s)}}{1 + N(a)}$$

其中 $\mathcal{A}(s)$ 是从补全里提取到的候选动作集合。分母由 UCT 的 $\sqrt{N(a)}$
变成 $1 + N(a)$（线性衰减），因为已有先验引导，可以更激进地收敛。

| | 纯 MCTS | 动态词表 | MolSEDD |
|---|:--:|:--:|:--:|
| 候选动作 | 固定全词表 | SEDD 补全 | SEDD 补全 |
| 选择公式 | UCT | UCT | **PUCT + 先验** |
| SEDD 参与排序 | 否 | 否 | **是** |
| 探索项衰减 | $\sqrt{N(a)}$ | $\sqrt{N(a)}$ | $1 + N(a)$ |

---

## 3. 候选动作与先验是怎么来的

对一个节点（当前前缀 $s$），把 $s$ 钉住采样 `discovery_bs` 个分子，
然后逐条处理：

```python
for smi in smiles:
    if validity_filter and Chem.MolFromSmiles(smi) is None:
        continue                              # 1. 可选：丢弃解析不了的分子
    toks = spec.tokenize(smi)
    if len(toks) <= len(prefix) or toks[:len(prefix)] != prefix:
        continue                              # 2. 必须比前缀长，且前几位对得上
    if toks[len(prefix)] in vocabulary:
        counter[toks[len(prefix)]] += 1       # 3. 取前缀后的第一个 token 计数
```

于是：

- 候选集合 $\mathcal{A}(s)$ = `counter` 的键；
- `prior="sedd"` 时 $P(s,a) = C(s,a)/\sum C$，`prior="uniform"` 时每个候选取 $1/|\mathcal{A}(s)|$；
- 若 `dynamic_vocab=False`（纯 MCTS），候选集合直接取整个词表，先验均匀。

先验是**经验频率归一化**，不是模型 softmax 输出。

---

## 4. 树节点、动作与终止

**节点**就是一段 SMILES 前缀（token 列表），例如 `['N', 'C', 'O', '(']` → `"NCO("`。
根节点是空前缀。扩散的中间状态只存在于采样过程内部，和树节点无关。

**动作**是在前缀末尾追加一个 token：

```python
child = MCTSNode(node.tokens + [action], parent=node, prior=prior)
```

**深度上限**由 `max_steps` 控制，选择循环在 `len(node.tokens) >= max_steps` 时停止。
各数据集的默认值见对应 `run_*.py` 里的 `DATASET_CONFIG`。

**死亡节点**：奖励为 0 且整批平均 LogP ≤ −90（说明这批分子基本没生成出来），
标记 `is_dead` 并在后续选择里跳过。

**首层**：根节点的候选动作被限制在该数据集的起始 token 集合内（如 MOSES 的
`Br, C, Cl, F, N, O, [H], c`），避免从一个无法延伸出合法分子的 token 起步。

---

## 5. 数值例子

设目标 $y^* = 1$，某节点采样出 5 个有效分子，LogP 为 `[0.8, 1.2, 0.5, 1.5, 1.0]`：

```
MAE  = (0.2 + 0.2 + 0.5 + 0.5 + 0.0) / 5 = 0.28
R    = exp(-0.28) ≈ 0.756
```

于是该节点回传奖励 0.756；若它的子动作被选中 $n$ 次、累积奖励 $V$，则
$Q = V/n$，再按 2.1 / 2.3 的公式加上探索项参与下一轮选择。
