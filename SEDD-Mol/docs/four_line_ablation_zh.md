# 折线图四条线的公式说明

| 线 | 词表 | 有效性检测 | 先验 prior | 公式 |
|---|---|---|---|---|
| Ours | 动态词表 | 有 | 频率先验 | PUCT |
| MCTS + dynamic vocab + validity | 动态词表 | 有 | 无（均匀） | UCT |
| MCTS + dynamic vocab | 动态词表 | 无 | 无（均匀） | UCT |
| Pure MCTS | 静态词表 | 无 | 无（均匀） | UCT |

## 公式

### UCT（下面三条线使用）

$$
Q(a) \;+\; c \cdot \sqrt{\frac{2\ln N_{parent}}{n(a)}}
$$

### PUCT（Ours 使用）

$$
Q(a) \;+\; c \cdot P(a) \cdot \frac{\sqrt{N_{parent}}}{1 + n(a)}
$$

其中：

- $Q(a)$：子节点 $a$ 的平均价值
- $N_{parent}$：父节点访问次数
- $n(a)$：子节点 $a$ 的访问次数
- $c$：探索常数
- $P(a)$：子节点 $a$ 的先验概率（SEDD 生成频率）
