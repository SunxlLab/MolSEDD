# MCTS 三种方法的公式详解

## 基础符号定义

| 符号 | 含义 | 代码变量 |
|------|------|----------|
| $V(a)$ | 动作 $a$ 的累积奖励总和 | `child.value` |
| $N(a)$ | 动作 $a$ 被访问的次数 | `child.visits` |
| $Q(a)$ | $Q(a) = V(a) / N(a)$，动作 $a$ 的历史平均奖励 | `child.value / child.visits` |
| $N_{\text{parent}}$ | 父节点的总访问次数 | `self.visits` |
| $c$ | 探索常数（权衡"利用已知好动作"与"尝试未知动作"） | `c_param`（默认 1.414 或 0.618） |
| $P_{\text{SEDD}}(s,a)$ | SEDD 对动作 $a$ 的生成频率归一化（方案A 专属） | `child.prior` |

---

## 奖励 $R$ 的计算（三种方法共用）

三种方法的奖励函数完全一致，均基于 LogP 的 MAE（平均绝对误差）：

```python
mae_logp = Σ|target_logp - logp_i| / valid_count
reward = exp(-mae_logp)
```

$$R = e^{-\text{MAE}}, \quad \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - y^* \right|$$

- $y_i$：生成分子的 LogP 值
- $y^*$：目标 LogP 值（默认 1.0）
- $R \in [0, 1]$，越接近目标 → MAE 越小 → reward 越接近 1

**若有效率低于 50%**（前缀太差，大部分分子无效），reward 直接设为 0，该节点标记为死亡。

---

## 方法 1：纯 MCTS — UCT（Undetermined Confidence Bounds for Trees）

**文件**：`MOSEScond_PureMCTS.py`

**选择公式**（第 168-173 行）：

```python
(child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
```

$$\text{Score}(a) = \underbrace{\frac{V(a)}{N(a)}}_{Q(a):\ \text{利用项}} \;+\; c \cdot \underbrace{\sqrt{\frac{2 \cdot \ln N_{\text{parent}}}{N(a)}}}_{\text{探索项}}$$

| 项 | 作用 |
|----|------|
| 利用项 $Q(a)$ | 越大越好：历史上奖励高的动作被优先选择 |
| 探索项 | 分母 $N(a)$：被访问越少，探索项越大（鼓励尝试未知动作）。分子 $\ln N_{\text{parent}}$：父节点访问越多，越信任 $Q(a)$ 而减少探索 |

**候选动作来源**：固定 27 个 token 的全词表，所有节点共享。

**SEDD 参与程度**：仅用于 Simulation 阶段（评估前缀好坏），不参与子节点选择。

---

## 方法 2：GGTMCTS — UCT + 动态词表

**文件**：`MOSEScond_GGTMCTS.py`

**选择公式**（第 163-165 行，**与方法 1 完全相同**）：

$$\text{Score}(a) = \frac{V(a)}{N(a)} + c \cdot \sqrt{\frac{2 \cdot \ln N_{\text{parent}}}{N(a)}}$$

**候选动作来源**：`discover_node_children()` 每轮生成 200 个分子，从有效分子中提取实际出现的下一个 token 作为候选集。SEDD 参与候选发现，但不参与子节点权重排序。

**SEDD 参与程度**：候选发现 + 模拟评估，但搜索策略无 SEDD 偏好。

---

## 方法 3：SEDD-weighted MCTS — PUCT（Predictor + UCT）

**文件**：`MOSEScond_SEDDweighted_MCTS.py`

**选择公式**（第 197-199 行）：

```python
(child.value / child.visits) + c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
```

$$\text{Score}(a) = \underbrace{\frac{V(a)}{N(a)}}_{Q(a):\ \text{利用项}} \;+\; c \cdot \underbrace{P_{\text{SEDD}}(s,a)}_{\text{先验}} \cdot \underbrace{\frac{\sqrt{N_{\text{parent}}}}{1 + N(a)}}_{\text{探索衰减项}}$$

| 项 | 作用 |
|----|------|
| 利用项 $Q(a)$ | 同上 |
| 先验 $P_{\text{SEDD}}(s,a)$ | SEDD 生成频率归一化：`count(a) / total_count`。SEDD 偏好的 token 频率高 → prior 大 → 探索权重高 → 优先被选中 |
| 探索衰减 $\frac{\sqrt{N_{\text{parent}}}}{1+N(a)}$ | 分母 $1+N(a)$ 线性衰减（比 UCT 的 $\sqrt{N(a)}$ 更快），因为已有 prior 引导，可以更激进剪枝 |

**SEDD prior 计算方式**（`discover_node_children` 第 234-238 行）：

```python
next_token_counter = Counter()          # 统计每个候选 token 的出现次数
# ... 遍历 200 个生成分子 ...
total_count = sum(next_token_counter.values())
node.sedd_priors = {token: count / total_count for token, count in next_token_counter.items()}
```

$$P_{\text{SEDD}}(s, a) = \frac{\text{count}(a)}{\sum_{k} \text{count}(a_k)}$$

**SEDD 参与程度**：候选发现 + 模拟评估 + **PUCT 先验引导搜索方向**。

---

## 三方法对照总结

| | 纯 MCTS | GGTMCTS | SEDD-weighted |
|------|:--:|:--:|:--:|
| 候选动作来源 | 固定 27 token | SEDD 动态发现 | SEDD 动态发现 |
| 选择公式 | UCT | UCT | **PUCT + SEDD prior** |
| 是否存在先验 | 无 | 无 | $P_{\text{SEDD}}(s,a)$ |
| SEDD 参与搜索 | ❌ | ❌ | **✅** |
| 探索项衰减速度 | 慢（$\sqrt{n}$） | 慢（$\sqrt{n}$） | 快（$1+n$） |

---

---

## 树节点、动作与候选集合的实现细节（依据代码）

### 1. Tree Node 表示什么

**节点 = 部分 SMILES 前缀（token 列表）**，不是完整分子，也不是扩散中间状态。

```python
# 三份代码一致
self.tokens = tokens   # 例如 ['N', 'C', 'O', '('] → "NCO("
```

- 根节点：`tokens = []`（空前缀）
- 每个节点的状态就是当前已拼接的前缀字符串
- 扩散中间状态（带噪声的 token ID 序列）**只存在于 `generate()` 内部的采样过程中**，与树节点无关

### 2. Action 是什么

**Action = 在前缀末尾追加（append）一个 token**，不是替换，不是扩散跃迁。

```python
# PureMCTS 第 276-279 行
action = node.untried_actions.pop()
new_tokens = node.tokens + [action]     # 追加一个 token
new_node = MCTSNode(new_tokens, parent=node)
```

```python
# GGTMCTS / SEDDweighted 同样
new_tokens = node.tokens + [action]
new_node = MCTSNode(new_tokens, parent=node, prior=action_prior)
```

### 3. 候选集合如何从 SEDD 输出产生

| 方法 | 候选来源 |
|------|----------|
| 纯 MCTS | 固定 27 个 token 全词表（`tok_dict_list.copy()`），人工指定根节点合法首字符 |
| GGTMCTS | `discover_node_children()`：SEDD 生成 `batch_size=200` 个补全分子，**仅从 RDKit 验证有效的分子**中提取 `tokens[current_len]`（前缀后第一个 token）作为候选 |
| SEDD-weighted | 同上 + Counter 统计频率 |

候选产生的具体代码（GGTMCTS 第 188-211 行 / SEDDweighted 第 238-256 行）：

```python
for smi in smiles_list:
    if Chem.MolFromSmiles(smi) is None:     # 1. RDKit 有效性过滤
        continue
    tokens = char_tokenizer(smi)
    if len(tokens) > current_len:           # 2. 分子必须比前缀长
        match = 前缀逐 token 匹配检查        # 3. 前几位必须与 node.tokens 一致
        if match:
            next_token = tokens[current_len]  # 4. 取前缀后第一个 token
            if next_token in tok_dict_list:   # 5. 必须在词表内
                next_token_counter[next_token] += 1   # 计数（SEDDweighted）
```

### 4. Transition Score 如何归一化

仅 SEDD-weighted 有归一化步骤（第 232-260 行）：

```python
total_count = sum(next_token_counter.values())   # 总出现次数
node.sedd_priors = {token: count / total_count for token, count in next_token_counter.items()}
```

$$P_{\text{SEDD}}(s,a) = \frac{\text{count}(a)}{\sum_k \text{count}(a_k)}$$

- **只在候选集合内归一化**（被 SEDD 生成出来且有效的 token 才参与）
- 若 `total_count = 0`（没有任何有效候选），回退为均匀分布 `1/K`
- 注意：这是**经验频率归一化**，不是模型 softmax 输出

### 5. No-prior / Uniform-weighted / SEDD-weighted 的确切选择公式

| 变体 | 公式 | 代码位置 |
|------|------|----------|
| **No-prior**（纯 MCTS 和 GGTMCTS） | $Q(a) + c\sqrt{\frac{2\ln N_{\text{parent}}}{N(a)}}$ | PureMCTS 168-173 / GGTMCTS 163-165 |
| **Uniform-weighted**（PUCT，$P=\frac{1}{K}$） | $Q(a) + c \cdot \frac{1}{K} \cdot \frac{\sqrt{N_{\text{parent}}}}{1+N(a)}$ | 消融文档中的理论对照，当前代码未单独实现（GGTMCTS 实际用 No-prior UCT） |
| **SEDD-weighted** | $Q(a) + c \cdot P_{\text{SEDD}}(s,a) \cdot \frac{\sqrt{N_{\text{parent}}}}{1+N(a)}$ | SEDDweighted 197-199 |

> 注意：代码中"GGTMCTS"虽然对应消融文档的"Uniform-prior"组，但它的选择公式实际是 No-prior 的 UCT（没有乘 $1/K$ 先验项），仅通过动态词表与纯 MCTS 区分。

### 6. Terminal State 与搜索深度

**深度限制**：

```python
# 三份代码一致：MOSES total_len=47，留 1 位给结束符 '$'，所以 max_steps=46
max_steps = 46
```

- 选择循环在 `len(node.tokens) >= max_steps` 时停止（前缀已达 46 个 token）
- 47 位中的最后 1 位由 suffix `$`（End token）固定

**死亡节点（Dead Node）**：

```python
# GGTMCTS: reward==0 且 avg_logp<=-90 → 标记死亡（有效率 < 50% 或 LogP 计算失败）
if reward == 0.0 and avg_logp <= -90:
    node.is_dead = True

# PureMCTS: reward==0 → 标记死亡
if reward == 0.0:
    node.is_dead = True
```

- 死亡节点的祖先回溯会跳过它（`valid_children = [c for c in children if not c.is_dead]`）
- 根节点 `visits=1` 初始化，子节点访问数从 0 开始

**结果路径提取**：搜索结束后沿"访问次数最多"（Robust Child）或"平均奖励最高"（Max Reward）两条路径回溯得到最终前缀。

---

## PUCT 与 UCT 数值对比示例

条件：$N_{\text{parent}}=100$，两个子节点 A、B 各访问 10 次，$Q=0.5$ 相同。SEDD prior：A=0.6 更受偏好，B=0.2 较冷门：

```
UCT（方法1/2）:
  A: 0.5 + 1.414 × √(2·ln(100)/10) = 0.5 + 1.414 × 0.960 = 1.858
  B: 0.5 + 1.414 × √(2·ln(100)/10) = 0.5 + 1.414 × 0.960 = 1.858  ← 完全相同，随机选

PUCT（方法3）:
  A: 0.5 + 0.618 × 0.6 × √100 / (1+10) = 0.5 + 0.618 × 0.6 × 10/11 = 0.837  ← 优先选中
  B: 0.5 + 0.618 × 0.2 × √100 / (1+10) = 0.5 + 0.618 × 0.2 × 10/11 = 0.612
```

> SEDD prior 使搜索偏向模型认为更合理的化学路径，从而更快收敛到目标 LogP。
