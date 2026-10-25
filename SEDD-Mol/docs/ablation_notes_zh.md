
假设只允许补充**少量、快速、但能明显提升论文说服力的实验**，我不会优先增加新的性质目标、更多数据集或复杂baseline，而会补一组非常集中的：

# 首选：预算匹配的条件生成机制对照实验

用同一个预训练SEDD模型、同一组LogP目标和同一套MCTS参数，比较三种条件生成方式：

### A. MolSEDD：SEDD prior + MCTS

即当前方法：

$$
P(s,a)=P_{\mathrm{SEDD}}(s,a)
$$

作为PUCT中的prior。

### B. Uniform-prior MCTS

保持以下内容完全相同：

-   相同SEDD模型；
    
-   相同候选动作集合；
    
-   相同reward；
    
-   相同搜索深度；
    
-   相同MCTS预算。
    

只把候选动作的prior改成均匀分布：

$$
P(s,a)=\frac{1}{|\mathcal A(s)|}
$$

这可以隔离验证：

> SEDD给出的transition information是否真的帮助了MCTS，而不是只要有MCTS就可以。

### C. 无条件SEDD采样＋事后筛选

直接由SEDD无条件生成相同预算数量的候选分子，计算LogP后选出距离目标最近的分子，不使用MCTS。

这可以回答：

> 为什么需要MCTS？是不是直接多生成一些分子再筛选，就能达到相同效果？

当前稿件已经展示了SEDD无条件生成和SEDD+MCTS条件生成，但现有消融只比较了Sentence Packing/Single Sequence及Uniform/Absorbing训练配置，没有隔离MCTS本身和SEDD prior本身的贡献。

MolSEDD

这个对照非常重要，因为已有分子MCTS方法如ChemTS和ParetoDrug本身就使用学习到的生成模型概率引导树搜索；TreeG等training-free diffusion guidance工作也将“候选生成”和“候选评价”作为需要分别验证的组成部分。审稿人很自然会追问：你们的MCTS是否优于简单筛选，SEDD prior是否优于无信息prior。[![](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov&sz=128)PMC+2![](https://www.google.com/s2/favicons?domain=https://www.nature.com&sz=128)Nature+2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5801530/?utm_source=chatgpt.com)

---

# 一、这组实验为什么最能让论文“上一个台阶”

它可以同时回答当前论文最关键的两个问题。

## 问题1：MCTS是否真的有必要

比较：

$$
\text{SEDD+MCTS} \quad \text{vs.} \quad \text{SEDD sampling+screening}
$$

假如在相同LogP计算次数下，MCTS能够：

-   更快接近目标；
    
-   获得更低的目标MAE；
    
-   获得更高的目标达成率；
    

就可以证明MCTS不仅是一个形式上的附加模块，而是比简单扩大无条件采样量更有效的目标搜索方法。

## 问题2：SEDD prior是否真的有作用

比较：

$$
\text{SEDD-prior MCTS} \quad \text{vs.} \quad \text{Uniform-prior MCTS}
$$

假如SEDD prior能够在相同搜索预算下：

-   更快收敛；
    
-   更准确达到目标LogP；
    
-   或者在达到相同LogP时保持更好的Validity、SA、RA和结构多样性；
    

那么论文才有依据把“SEDD-derived prior”提升为核心方法贡献。

这比再增加一个QED、TPSA或MW条件生成实验更有价值。增加性质目标只能证明“还能控制另一个性质”，而这个对照能够证明：

> **为什么你们当前的方法设计是必要的。**

---

# 二、如何做到公平比较

公平比较的关键不是统一运行时间，而是统一**性质函数调用预算**。

因为LogP计算虽然很快，但在更一般的目标中，reward可能来自：

-   分子对接；
    
-   预测模型；
    
-   合成评价；
    
-   实验代理模型。
    

因此建议令三种方法具有相同的property evaluation budget，例如每次生成任务均允许计算10,000次LogP。

具体控制：

-   同一个SEDD checkpoint；
    
-   相同的LogP目标；
    
-   相同的最终输出分子数量；
    
-   相同的property evaluation次数；
    
-   相同随机种子；
    
-   MCTS两组使用相同候选动作集合；
    
-   Uniform-prior组只替换动作权重，不改变候选生成方式。
    

最后一点非常重要。假如SEDD仍负责产生候选动作，则Uniform-prior实验检验的是：

> SEDD概率排序是否有额外价值。

而不是同时把候选生成机制也更换掉。

---

# 三、建议使用哪些数据和目标

为了兼顾说服力和计算成本，我会这样安排：

## 主要实验：MOSES

使用现有三个目标：

-   LogP=1；
    
-   LogP=2；
    
-   LogP=3。
    

MOSES规模较大、结构更接近lead-like molecules，也是论文中条件生成最有应用意义的数据集。

## 辅助验证：QM9

不一定需要把−1、0、1三个目标全部重做。可以选择距离原始分布较远、控制难度较大的一个目标，例如：

-   LogP=−1。
    

这样可以证明结论不是MOSES上的偶然现象，又不会显著扩大计算量。

不过，如果现有代码批量运行成本很低，把QM9三个目标全部跑完会使结果更整齐。

---

# 四、必须报告的核心指标

## 1\. 目标控制效果

至少报告：

### 目标误差

$$
\mathrm{MAE}_{\mathrm{target}} = \frac{1}{N} \sum_{i=1}^{N}|y_i-y^*|
$$

其中 $y^*$ 是目标LogP。

### 目标达成率

例如：

$$
\mathrm{Success@0.2} = \frac{\#\{|y_i-y^*|\leq0.2\}}{N}
$$

可以同时报告：

-   Success@0.1；
    
-   Success@0.2。
    

MAE反映整体接近程度，Success rate更容易直观比较。

## 2\. 搜索效率

画一条：

> best-so-far MAE或平均MAE随property evaluation次数变化的曲线。

横轴：

-   LogP evaluations；
    
-   或MCTS iterations。
    

纵轴：

-   target MAE；
    
-   或success rate。
    

这张图可能比最终结果表更能说明SEDD prior的价值。假如SEDD-prior组在较少评估次数时就达到较低误差，就可以合理表达：

> SEDD-derived prior improves the efficiency of property-directed search.

这条结论不需要声明它是“最优prior”，但已经足以明显强化论文。

---

# 五、条件生成质量指标

你之前提出补充：

-   Validity；
    
-   Uniqueness；
    
-   Novelty。
    

这三项应该加入，而且三种方法都应统一计算。

但我建议再增加两个几乎不需要重新生成样本的指标：

## 1\. Internal Diversity

基于Morgan fingerprint计算生成分子之间的平均不相似度，例如：

$$
\mathrm{IntDiv} = 1-\frac{1}{N(N-1)} \sum_{i\ne j} \mathrm{Tanimoto}(f_i,f_j)
$$

它能够发现这样一种情况：

> 分子虽然都不完全重复，Uniqueness很高，但彼此只相差一个原子或一个取代基。

MOSES基准本身就将Internal Diversity用于评价生成集合内部的化学多样性，因此加入这一指标十分自然。[![](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov&sz=128)PMC+1](https://pmc.ncbi.nlm.nih.gov/articles/PMC7775580/?utm_source=chatgpt.com)

## 2\. Scaffold diversity

可以报告：

-   unique Bemis–Murcko scaffold比例；
    
-   scaffold数量；
    
-   或scaffold diversity。
    

这可以进一步区分：

-   许多不同分子来自同一个骨架；
    
-   还是模型确实产生了多个不同骨架。
    

计算成本很低，基本属于对已有生成结果的后处理。

---

# 六、最终结果表建议

可以做成一张表：

| Method | Target MAE↓ | Success@0.2↑ | Validity↑ | Uniqueness↑ | Novelty↑ | IntDiv↑ | Unique scaffolds↑ | SA↓ | RA↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEDD sampling + filtering |  |  |  |  |  |  |  |  |  |
| Uniform-prior MCTS |  |  |  |  |  |  |  |  |  |
| SEDD-prior MCTS |  |  |  |  |  |  |  |  |  |

各目标可以分别报告，也可以再给一个Across-target average。

至少重复3个随机种子，报告：

$$
\text{mean}\pm\text{standard deviation}
$$

计算资源允许时用5个种子更稳妥。离散扩散条件引导工作也通常同时评价目标性质、Validity、Novelty，并通过多随机种子报告均值和标准差。[![](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov&sz=128)PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12989473/?utm_source=chatgpt.com)

---

# 七、不同结果分别意味着什么

这组实验的价值还在于：**无论结果怎样，都能帮我们确定论文应当讲什么故事。**

## 情况1：SEDD prior明显优于Uniform prior

例如：

-   更低target MAE；
    
-   更高success rate；
    
-   更快收敛；
    
-   同时保持相近或更好的生成质量。
    

那么可以把论文中心提升为：

> SEDD transition information provides an effective model-informed prior for MCTS-based molecular steering.

此时transition-ratio prior可以成为真正的核心贡献，而不只是实现细节。

## 情况2：两种MCTS接近，但均优于事后筛选

那么结论是：

> MCTS有效，但尚不能证明SEDD prior的独立优势。

论文主线仍应保持：

> SEDD用于分子生成，并通过MCTS扩展到条件生成。

不要重点宣称transition ratios是关键机制。

## 情况3：SEDD prior只提高收敛速度，最终结果相近

这仍然是有价值的结果。

可以写：

> SEDD-derived priors improve search efficiency under limited evaluation budgets.

对于真实药物设计中昂贵的reward函数，这种优势是有意义的。

## 情况4：MCTS与无条件筛选效果接近

那么“为什么需要MCTS”会比较难回答。论文可以保留条件生成功能展示，但不能把MCTS写成明显优于简单搜索的方法。

## 情况5：目标控制改善，但IntDiv或scaffold diversity明显下降

这不等于实验失败，而说明存在：

> property accuracy–structural diversity trade-off。

此时应如实收窄“without sacrificing diversity”之类的结论。

---

# 八、可选的第四组：Shuffled-prior MCTS

假如代码修改非常简单，可以增加一个几乎免费的组：

> 保留SEDD prior的数值分布，但在候选动作间随机打乱prior。

它可以区分：

-   非均匀prior本身带来的效果；
    
-   SEDD给出的正确动作排序带来的效果。
    

但它不是必需项。三组实验已经能够形成清楚的逻辑闭环，不必为了消融完整性让实验变得过于复杂。

---

# 九、我不会优先补充的实验

在时间有限的情况下，不建议优先做：

-   新增QED、TPSA或MW条件目标；
    
-   多目标优化；
    
-   新增大批外部生成模型baseline；
    
-   大规模超参数搜索；
    
-   条件生成与完整训练集的t-SNE重合；
    
-   新的数据集；
    
-   湿实验或分子对接。
    

这些实验会扩展论文范围，却不直接修复当前最核心的证据缺口。

## 最终优先级

最值得补充的是一组紧凑的三臂实验：

$$
\boxed{ \text{SEDD sampling+filtering} \quad\text{vs.}\quad \text{Uniform-prior MCTS} \quad\text{vs.}\quad \text{SEDD-prior MCTS} }
$$

再统一报告：

$$
\boxed{ \text{target accuracy} + \text{evaluation efficiency} + \text{Validity/Uniqueness/Novelty} + \text{IntDiv/scaffold diversity} + \text{SA/RA} }
$$

这不是简单“再补几个指标”，而是用一次小型实验同时验证：

> **为什么使用MCTS，以及为什么让SEDD参与MCTS。**

它比新增多个条件性质更可能使论文的方法贡献和实验结论真正对齐。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://www.nature.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov&sz=32)



这里确实容易混淆，因为论文里同时出现了两个完全不同的“Uniform”。

## 先区分两个概念

### 1\. S-U中的Uniform transition

这是**SEDD训练阶段的扩散转移矩阵**：

-   S：Single Sequence；
    
-   U：Uniform diffusion transition matrix。
    

它决定前向扩散如何对SMILES token加噪，以及SEDD如何学习反向过程。论文最终选择的S-U就是用这种Uniform转移矩阵训练出来的SEDD模型。

MolSEDD

### 2\. Uniform prior in MCTS

这是**模型训练完成后，MCTS搜索阶段的PUCT动作先验**。

它与SEDD训练时采用Uniform还是Absorbing转移矩阵不是一回事，也不需要重新训练SEDD。

更准确地说，我前面应该称它为：

> **Uniformized PUCT prior over SEDD-proposed candidates**

即“对SEDD提出的候选动作使用均匀的PUCT先验”。

---

# 当前MolSEDD怎样使用SEDD prior

假设在某个MCTS节点 $s$，SEDD给出 $K$ 个候选动作：

$$
\mathcal A_{\mathrm{SEDD}}(s) = \{a_1,a_2,\ldots,a_K\}
$$

同时SEDD为这些候选动作给出transition score或transition ratio：

$$
r_1,r_2,\ldots,r_K
$$

你们当前方法大致是把这些值归一化：

$$
P_{\mathrm{SEDD}}(s,a_i) = \frac{r_i}{\sum_{j=1}^{K}r_j}
$$

然后放入PUCT：

$$
\operatorname{PUCT}(s,a) = Q(s,a) + cP(s,a) \frac{\sqrt{N(s)}}{1+N(s,a)}
$$

因此，SEDD认为更可能的动作会在搜索早期获得更高的探索优先级。论文当前方法部分确实把SEDD同时描述为候选生成来源和MCTS prior来源。

MolSEDD

---

# Uniform-prior对照如何保持同一个SEDD模型

对照实验中，**SEDD模型、checkpoint和候选动作集合全部不变**。

仍然由同一个SEDD模型产生：

$$
\mathcal A_{\mathrm{SEDD}}(s) = \{a_1,\ldots,a_K\}
$$

但是不再使用SEDD给出的相对分数决定PUCT优先级，而是在这些候选中赋予相同权重：

$$
P_{\mathrm{uniform}}(s,a_i) = \frac{1}{K}
$$

所以两组的区别只有一处：

| 组成部分 | SEDD-prior MCTS | Uniformized-prior MCTS |
| --- | --- | --- |
| SEDD checkpoint | 相同 | 相同 |
| SEDD训练方式 | 相同S-U | 相同S-U |
| 候选动作 | 相同 | 相同 |
| 候选数量 $K$ | 相同 | 相同 |
| MCTS reward | 相同 | 相同 |
| 搜索预算 | 相同 | 相同 |
| PUCT中的 $P(s,a)$ | 按SEDD分数归一化 | 全部设为 $1/K$ |

因此不需要重新训练，只需要在MCTS推理代码中改变一行类似的逻辑：

```
Python

Run

```
# 当前方法prior=sedd_scores/sedd_scores.sum()

# Uniformized prior对照prior=torch.ones_like(sedd_scores) /len(sedd_scores)
```
```

---

# 这个实验实际检验什么

它检验的不是：

> SEDD模型有没有用。

因为两组都使用了SEDD模型产生候选动作。

它检验的是：

> **在候选动作已经由SEDD确定的情况下，SEDD输出的相对概率排序是否能进一步改善MCTS搜索。**

也就是说，它只隔离“prior weighting”的作用：

$$
\text{SEDD候选集合} + \boxed{\text{SEDD相对权重}}
$$

对比：

$$
\text{相同SEDD候选集合} + \boxed{\text{均匀权重}}
$$

假如SEDD-prior组更快达到目标或具有更好的最终结果，就说明SEDD的相对transition information不仅能提出候选，还能有效安排搜索优先级。

---

# 这个对照仍然有一个重要限制

因为Uniformized-prior组的候选动作仍由SEDD提出，所以它不能回答：

> 使用SEDD提出候选本身有没有价值？

它只能回答：

> 对这些候选使用SEDD权重，是否优于均匀权重？

这是一个很干净、成本也很低的消融。

若要彻底去除SEDD影响，则需要令MCTS从：

-   整个token vocabulary；
    
-   随机候选；
    
-   或所有合法动作；
    

中搜索。但这样候选集合和分支数量都会变化，比较不再公平，而且计算量可能大幅增加。因此不适合作为第一优先的快速实验。

---

# 与论文中S-U消融的关系

这两组消融位于不同层级：

## 现有S-U消融：训练层

检验的是：

> 什么样的SEDD训练设置更适合无条件分子生成？

比较：

-   Sentence Packing与Single Sequence；
    
-   Absorbing与Uniform diffusion transition。
    

## 新的prior消融：搜索层

检验的是：

> 训练好的SEDD输出的相对transition scores，是否适合作为MCTS的动作优先级？

比较：

-   SEDD-weighted PUCT；
    
-   Uniform-weighted PUCT。
    

所以这两个“Uniform”没有直接对应关系。

---

# 最严谨的实验命名

为了避免审稿人产生与你相同的疑问，论文中不要简单写：

> Uniform prior

建议写为：

> **Uniformized PUCT prior over identical SEDD-proposed actions**

或者简写：

-   **SEDD-weighted MCTS**
    
-   **Uniform-weighted MCTS**
    

并明确说明：

> Both variants use the same pretrained SEDD checkpoint and the same SEDD-proposed candidate actions. They differ only in whether the PUCT prior preserves or removes the relative scores assigned by SEDD.

因此，这项实验**不需要重新训练模型**，只需要确认当前代码中“候选动作的产生”和“候选动作的PUCT权重”是否可以拆开。假如当前实现已经先得到候选集合及对应概率，再进入PUCT，那么改动会非常小。

Sources


下面这份可以直接作为补充实验任务说明。

---

# MolSEDD补充实验任务说明

## ——验证MCTS的必要性及SEDD搜索先验的实际作用

目前论文已经完成两部分主要工作：

1.  证明SEDD可以用于SMILES分子无条件生成；
    
2.  在训练好的SEDD模型基础上加入MCTS，实现目标LogP条件生成。
    

现有结果已经能够说明MolSEDD整体框架可以工作，但还没有回答两个审稿人很可能提出的问题：

> 第一，为什么一定需要MCTS？直接使用SEDD大量采样后筛选是否也能达到类似效果？  
> 第二，MCTS中使用SEDD提供的transition information作为动作先验，是否真的比不使用这些相对权重更有效？

当前论文的消融实验只比较了Sentence Packing/Single Sequence和Absorbing/Uniform diffusion transition，属于**SEDD训练层面的消融**，尚未对MCTS搜索阶段进行机制验证。

MolSEDD

因此，本次补充实验的目的不是增加更多数据集或更多目标性质，而是用一个规模较小、变量控制清楚的实验，验证：

> **MCTS是否比简单采样筛选更有效，以及SEDD输出的相对transition scores是否能为MCTS提供有价值的搜索优先级。**

---

# 一、实验开始前需要先检查代码实现

在正式运行实验之前，先梳理当前MCTS代码，明确以下内容：

1.  每个MCTS节点的候选动作是如何产生的；
    
2.  SEDD输出的是完整词表上的score、Top-k候选，还是通过概率采样得到的候选；
    
3.  SEDD输出的transition score或transition ratio如何转换为PUCT中的 $P(s,a)$；
    
4.  候选集合的产生与候选权重的赋值是否可以分开；
    
5.  一次MCTS运行中实际调用多少次LogP计算函数。
    

最关键的是区分：

-   **候选动作集合由谁产生；**
    
-   **候选动作在PUCT中如何加权。**
    

本次实验需要尽量保证不同MCTS组使用**完全相同的候选动作集合**，只改变PUCT中的权重。否则无法判断性能差异究竟来自候选不同，还是来自prior weighting不同。

如果当前代码是先由SEDD生成Top-k候选及其score，再进入MCTS，那么实验修改比较简单。

如果当前代码是直接按照SEDD概率随机采样候选，则应尽量：

-   固定随机种子；
    
-   缓存每个节点产生的候选集合；
    
-   将同一候选集合分别送入两种MCTS策略。
    

不能一组使用SEDD高概率候选，另一组重新随机生成完全不同的候选后再进行比较。

---

# 二、需要设置的三组实验

## 组1：SEDD-weighted MCTS

这是当前MolSEDD的条件生成方法。

对于某一搜索状态 $s$，SEDD提供候选动作：

$$
\mathcal{A}(s)=\{a_1,a_2,\ldots,a_K\}
$$

并输出每个候选对应的transition score或transition ratio：

$$
r_1,r_2,\ldots,r_K
$$

将这些值归一化后构造PUCT prior：

$$
P_{\mathrm{SEDD}}(s,a_i) = \frac{r_i}{\sum_{j=1}^{K}r_j}
$$

然后按照当前论文中的PUCT公式进行搜索。

本组作为完整MolSEDD方法。

---

## 组2：Uniform-weighted MCTS over identical SEDD candidates

本组**不重新训练SEDD模型**。

必须使用与组1完全相同的：

-   SEDD checkpoint；
    
-   SEDD训练方式；
    
-   候选动作集合；
    
-   候选数量 $K$；
    
-   MCTS搜索深度；
    
-   exploration coefficient；
    
-   reward函数；
    
-   搜索预算；
    
-   随机种子。
    

唯一变化是：不再使用SEDD给出的相对score作为PUCT权重，而是对同一批候选动作赋予相同prior：

$$
P_{\mathrm{uniform}}(s,a_i) = \frac{1}{K}
$$

因此，两组的差异只能是：

| 设置 | SEDD-weighted MCTS | Uniform-weighted MCTS |
| --- | --- | --- |
| SEDD模型 | 相同 | 相同 |
| SEDD checkpoint | 相同 | 相同 |
| 候选动作 | 相同 | 相同 |
| 候选数量 | 相同 | 相同 |
| MCTS参数 | 相同 | 相同 |
| Reward | 相同 | 相同 |
| PUCT prior | 按SEDD score归一化 | 每个候选均为 $1/K$ |

本组实验检验的是：

> **在候选动作已经由SEDD确定的情况下，保留SEDD给出的相对概率排序，是否能进一步提高MCTS搜索效果。**

需要注意，本组中的“Uniform-weighted”与论文Table 5中的“Uniform transition matrix”不是同一个概念。

-   Table 5中的Uniform属于SEDD训练阶段的扩散转移矩阵；
    
-   本实验中的Uniform属于MCTS推理阶段的PUCT动作权重。
    

两者不能混用。

---

## 组3：Unconditional SEDD sampling + post hoc filtering

使用与前两组相同的预训练SEDD checkpoint，但不运行MCTS。

具体过程为：

1.  使用SEDD进行无条件分子采样；
    
2.  对所有生成分子计算LogP；
    
3.  按照与目标LogP的绝对误差排序；
    
4.  选择误差最小的前 $N$ 个分子作为最终结果。
    

本组检验的是：

> **MCTS是否比“多生成一些分子后再筛选”更有效。**

公平比较时，不能让本组无限制生成分子。三组应使用相同的property evaluation budget，即允许计算LogP的总次数相同。

例如：

-   SEDD-weighted MCTS：共进行10,000次LogP评价；
    
-   Uniform-weighted MCTS：共进行10,000次LogP评价；
    
-   SEDD sampling + filtering：生成并评价10,000个候选分子。
    

最终三组均输出相同数量的分子，例如1,000个。

---

# 三、数据集与目标设置

本次补充实验以MOSES为主，因为：

-   MOSES是当前论文中规模最大、最接近lead-like chemical space的数据集；
    
-   现有条件生成已经设置了LogP=1、2、3三个目标；
    
-   可以直接沿用现有模型和实验代码。
    

建议正式实验使用：

$$
\mathrm{LogP}_{\mathrm{target}}=1,\ 2,\ 3
$$

三个目标全部保留。

这样可以同时覆盖：

-   距离原始训练分布较远的目标；
    
-   位于训练分布中间区域的目标；
    
-   相对容易达到的目标。
    

至少使用3个独立随机种子，所有结果报告：

$$
\text{mean}\pm\text{standard deviation}
$$

如果计算量很小，可以使用5个随机种子。

QM9暂时不作为必做项。若MOSES结果较清楚、代码运行成本又很低，可以选择QM9中较困难的一个目标，例如LogP=-1，作为补充验证。

---

# 四、必须严格控制的变量

三组实验必须统一以下条件：

-   同一个SEDD checkpoint；
    
-   相同tokenizer；
    
-   相同最大序列长度；
    
-   相同目标LogP；
    
-   相同最终输出分子数；
    
-   相同LogP评价次数；
    
-   相同随机种子集合；
    
-   相同有效性判定方式；
    
-   相同去重和canonicalization规则。
    

两组MCTS还必须统一：

-   相同候选动作集合；
    
-   相同Top-k或branching factor；
    
-   相同PUCT coefficient；
    
-   相同MCTS iteration数；
    
-   相同search depth；
    
-   相同reward函数；
    
-   相同termination condition。
    

不能为了让某一组结果更好而单独调整超参数。

如果Uniform-weighted MCTS表现较差，也不要专门增加其搜索预算；否则实验将无法公平解释。

---

# 五、最重要的评价指标

## 1\. 目标性质误差

这是最重要的最终效果指标。

计算：

$$
\mathrm{Target\ MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| \mathrm{LogP}_i-\mathrm{LogP}_{\mathrm{target}} \right|
$$

Target MAE越低，说明目标控制越准确。

除均值外，建议同时报告：

-   median absolute error；
    
-   standard deviation；
    
-   最终LogP分布图。
    

---

## 2\. 目标达成率

定义目标容差范围，例如：

$$
\mathrm{Success@0.2} = \frac{ \#\left\{ \left|\mathrm{LogP}_i-\mathrm{LogP}_{\mathrm{target}}\right| \leq 0.2 \right\} }{N}
$$

建议报告：

-   Success@0.1；
    
-   Success@0.2。
    

Success rate比单独报告均值更直观，可以显示有多少生成分子真正达到目标附近。

---

## 3\. 搜索效率曲线

这是判断SEDD prior是否有意义的关键图。

横轴使用：

-   property evaluation次数；
    
-   或累计LogP函数调用次数。
    

纵轴可以选择：

-   当前最优target error；
    
-   当前Top-N平均MAE；
    
-   Success@0.2。
    

重点比较：

-   SEDD-weighted MCTS；
    
-   Uniform-weighted MCTS；
    
-   SEDD sampling + filtering。
    

如果SEDD-weighted组在相同评价次数下更快接近目标，则可以说明：

> SEDD-derived weighting improves search efficiency under a limited evaluation budget.

即使三组最终结果接近，只要SEDD-weighted组收敛更快，这一结果仍然有论文价值。

---

# 六、需要同时补充的生成质量指标

除目标控制外，三组都要统一计算：

-   Validity；
    
-   Uniqueness；
    
-   Novelty；
    
-   SA；
    
-   RA。
    

其中：

-   Validity用于判断条件搜索是否破坏分子有效性；
    
-   Uniqueness用于判断是否出现大量完全重复分子；
    
-   Novelty用于判断是否只是从训练集中调用已有结构；
    
-   SA和RA用于评价性质优化是否伴随明显的合成困难。
    

建议再补充两个只需要后处理、计算成本很低的指标：

## 1\. Internal Diversity

使用Morgan fingerprint和Tanimoto similarity计算生成集合内部多样性。

例如：

$$
\mathrm{IntDiv} = 1- \frac{1}{N(N-1)} \sum_{i\neq j} \mathrm{Tanimoto}(f_i,f_j)
$$

该指标用于识别一种情况：

> 虽然Uniqueness很高，但不同分子之间只存在很小的取代基差异。

## 2\. Scaffold diversity

使用Bemis–Murcko scaffold统计：

-   unique scaffold数量；
    
-   unique scaffold比例；
    
-   scaffold frequency distribution。
    

这个指标可以判断条件搜索是否集中到少量骨架。

本实验不要求条件生成覆盖整个MOSES训练集化学空间。性质优化本来就会把分布引导至目标LogP子空间。我们真正关注的是：

> 在目标性质区域内，是否仍能生成有效、新颖且不过度重复的不同结构。

---

# 七、最终应形成的结果表和图

## 表1：三个方法的最终性能比较

建议格式：

| Method | Target | MAE↓ | Success@0.1↑ | Success@0.2↑ | Validity↑ | Uniqueness↑ | Novelty↑ | IntDiv↑ | Unique scaffold ratio↑ | SA↓ | RA↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEDD sampling + filtering | 1 |  |  |  |  |  |  |  |  |  |  |
| Uniform-weighted MCTS | 1 |  |  |  |  |  |  |  |  |  |  |
| SEDD-weighted MCTS | 1 |  |  |  |  |  |  |  |  |  |  |

LogP=2和3同样列出。

所有结果报告mean±standard deviation。

## 图1：搜索效率曲线

每个目标分别绘制：

-   横轴：LogP evaluation budget；
    
-   纵轴：Target MAE或Success@0.2；
    
-   三条曲线对应三种方法。
    

## 图2：最终LogP分布

保留当前论文已有的KDE表达方式，将三种方法的结果放在同一目标下比较。

不需要制作过多结构示意图，先保证核心对照清楚。

---

# 八、最重要的结果是什么

这组实验最重要的不是某一个Validity数字，而是以下两个层级的结论。

## 第一优先级：SEDD weighting是否有用

比较：

$$
\text{SEDD-weighted MCTS} \quad \text{vs.} \quad \text{Uniform-weighted MCTS}
$$

理想结果是SEDD-weighted组在相同预算下表现出：

-   更低Target MAE；
    
-   更高Success rate；
    
-   更快收敛；
    
-   或在相同目标误差下具有更好的Validity、SA、RA和多样性。
    

只要其中一项表现稳定，特别是搜索效率明显提高，就可以支持：

> SEDD-derived transition information provides a useful model-informed prior for MCTS search.

但不要写成：

-   optimal prior；
    
-   guaranteed prior；
    
-   essential for all property optimization。
    

---

## 第二优先级：MCTS是否有必要

比较：

$$
\text{两种MCTS} \quad \text{vs.} \quad \text{SEDD sampling + filtering}
$$

理想结果是MCTS在相同LogP评价预算下：

-   更快达到目标；
    
-   Target MAE更低；
    
-   Success rate更高。
    

这可以证明MCTS不是简单增加了一个复杂模块，而是比无条件采样后筛选更有效。

---

# 九、不同实验结果应如何解释

本实验不要求结果必须符合预期，禁止只保留有利结果。

## 情况1：SEDD-weighted明显优于Uniform-weighted

可以将SEDD transition prior提升为论文的重要方法贡献。

## 情况2：两种MCTS接近，但均优于采样筛选

说明MCTS有效，但不能强调SEDD相对权重的独立贡献。

论文主线仍应是：

> SEDD用于分子生成，并通过MCTS扩展到条件生成。

## 情况3：SEDD-weighted收敛更快，但最终结果接近

仍然有价值，可以强调有限评价预算下的search efficiency。

## 情况4：MCTS与采样筛选接近

说明当前LogP任务可能较容易，不能强烈声称MCTS优于简单筛选。

此时应谨慎处理，不要人为增加筛选组难度。

## 情况5：目标误差降低，但Internal Diversity或scaffold diversity下降

说明存在：

> property accuracy–structural diversity trade-off。

应如实报告，不能再使用“without sacrificing diversity”等绝对表达。

---

## 本次补充实验的最终目标

本次实验不是为了证明MolSEDD全面优于所有方法，而是补齐当前论文最关键的内部证据链：

$$
\boxed{ \text{SEDD能够生成分子} \rightarrow \text{MCTS能够实现目标搜索} \rightarrow \text{SEDD提供的相对权重是否改善MCTS} }
$$

最理想的论文结论是：

> 在相同的预训练SEDD模型、候选动作和性质评价预算下，使用SEDD transition information进行加权的MCTS比均匀加权MCTS和无条件采样筛选更有效地接近目标LogP，同时保持较好的分子有效性、新颖性、多样性和合成可及性。

只有实验结果确实支持时，才能采用上述结论。

Sources