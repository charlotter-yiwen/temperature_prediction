# IEEE 论文写作体系

> 基于 57 篇 IEEE Thermal/PE 论文 + 芝加哥大学写作方法论 + IEEE 官方结构指南整合
> 阅读时间：2026-04-01
> 来源：IEEE TPEL Popular Articles、Thermal Journals、IEEE 官方指南、LaTeX 模板、芝加哥大学 Rubenstein 教授讲座

---

## 本体系的使用方法

当你开始写一篇 IEEE 论文时，按以下**四个阶段的顺序**工作：

```
第一阶段：准备 ──→ 第二阶段：起草 ──→ 第三阶段：完善 ──→ 第四阶段：修改
   (策略层)           (写作层)           (打磨层)           (检查层)
```

每个阶段都有明确的**输入**（上一阶段的产出）和**输出**（交给下一阶段的草稿）。

---

# 第一阶段：准备（写作之前做什么）

> **核心原则：写作不是传达你的想法——写作是改变读者的想法。**
> — Larry Rubenstein，芝加哥大学写作项目

大多数论文失败，不是因为写作水平差，而是因为**在动笔之前没有想清楚对读者有什么价值**。

---

## 1.1 理解你的读者（最关键步骤）

**芝加哥方法的起点：** 你不是要"传达你的想法"，而是要"让读者相信他们之前理解得不对"。

### 读者失败级联（当你写不好时）

```
你的写作模式 ≠ 读者的阅读模式
        ↓
第1步：读者减速（重读段落）
第2步：读者不理解
第3步：读者开始烦躁
第4步：读者停止阅读
```

**关键事实：** 审稿人是被迫读你的论文的（审稿职责）。但未来引用你论文的读者——他们只有在觉得有价值的情况下才会读。

### 价值层级（比"清晰"更重要的是"有价值"）

```
有价值（Value）     ← 没有它，其他都没用
    ↓
有说服力（Persuasion）
    ↓
有组织（Organization）
    ↓
清晰（Clarity）    ← 有它锦上添花，没它不是致命伤
```

**结论：** 不要先问"我的论文清晰吗？"——先问"**这篇论文对我的读者（同一领域的专家）有什么价值？**"

---

## 1.2 建立"你错了"结构（Introduction 的核心）

**这是所有顶级期刊论文 Introduction 的核心结构。**

**公式：**

```
你们（读者）一直做得很出色，
你们建立了 [已被广泛接受的模型/方法]，
但是——有一个小问题：[具体指出他们错了什么]。
本文将证明：[本文的解决方案/发现]。
```

**为什么有效：**
- 你没有侮辱读者
- 你是在**同一知识体系内**挑战他们
- 你在说"我很敬佩你们的工作，但这里有个细节有问题"
- 这激活了读者内心深处的"较真"欲望

**IEEE 热管理论文示例：**

```
[赞扬] Physics-informed neural networks (PINNs) have emerged as a promising
framework for modeling dynamical systems, with widespread adoption in
thermal analysis [X].

[挑战] However, existing PINN methods suffer from spectral bias when
applied to multi-scale thermal gradients in 3D-IC packages, leading to
significant errors at hotspots.

[本文方案] To address this issue, we propose [method], which introduces
[specific innovation]. As demonstrated in Section III, the proposed method
achieves MAE of 0.9°C, a 67% reduction compared with vanilla PINN.
```

---

## 1.3 社区密码：每个学术共同体都有自己的价值词汇

**每周 15 分钟练习（Larry 教授建议）：**

1. 选取你所在领域近 2 年的 1 篇顶级论文
2. 打印出来（纸质版）
3. 用荧光笔标出所有**为读者创造价值的词**（通常 5-10 个）
4. 记录这些词，形成你自己的价值词汇表

**不同学科的价值词汇：**

| 类别 | 价值词汇 | 作用 |
|------|---------|------|
| 挑战类 | however, but, although, nonetheless, yet | 暗示"现有理解有问题" |
| 问题类 | anomaly, inconsistency, gap, limitation, puzzle | 暗示"现有理论不完整" |
| 承认类 | widely accepted, reported, established | 承认读者群体的共识 |
| 新方案类 | proposed, introduced, demonstrates, reveals | 引入本文的贡献 |
| 比较类 | in contrast, unlike, differs from | 区分本文与现有工作 |

**IEEE 论文前两段必须有 10+ 个价值词汇。** 如果没有，读者会觉得"无价值"。

---

## 1.4 Argument（论证）vs. Explanation（解释）

**最常见的 PhD 写作者错误：用解释代替论证。**

**解释的逻辑（你被训练成这样）：**
> "你不懂是吗？那我来给你解释一下我脑子里在想什么。"

**为什么解释在学术写作中失败：**
> 没有人关心你脑子里在想什么。审稿人不需要你解释你的想法——他们需要你**预测他们会怀疑什么，然后解决那些怀疑**。

**Argument 的逻辑：**
> "我认为你们会对此产生疑问。你们可能会问：为什么？我的数据/理论表明：[解决他们的怀疑]。"

**Introduction 的本质不是解释，是"mini-argument"：** 给读者一个快速版本的理由，让他们相信"我需要继续读下去"。

---

## 1.5 认识你的读者会怀疑什么

**IEEE 论文读者的典型怀疑清单：**

| 读者可能怀疑的 | 你需要在论文中解决的 |
|-------------|-------------------|
| "这个方法在这个场景下能 work 吗？" | 广泛的实验验证（不同功率、组件数、温度条件）|
| "这个方法比现有方法真的更好吗？" | 与强 baseline 的公平比较 |
| "训练数据需求是否实际？" | 小样本、data-efficient 的证明 |
| "这个方法有理论保证吗？" | 消融实验 + 边界情况分析 |
| "我能复现这个方法吗？" | 公开代码、数据集、详细超参数 |
| "这对其他问题也适用吗？" | 跨应用领域的泛化性讨论 |

**知道读者怀疑什么 = 知道 Introduction 前两段应该说什么**

---

## 1.6 定位本文在文献中的位置

**Related Work 不是文献罗列，是批判性对话。**

**❌ AI 写法：**
```
"[X] proposed method A in 2020. [Y] proposed method B in 2021.
[Z] proposed method C in 2022."
```

**✅ 正确写法：**

```
Existing approaches for chip thermal analysis can be categorized into
two classes: physics-based simulation methods and data-driven methods.

Physics-based methods (e.g., HotSpot [1], 3D-ICE [2]) offer high accuracy
but require hours of computation, making them unsuitable for iterative
design exploration. Data-driven methods (e.g., U-Net [3], GAN [4])
significantly improve speed but at the cost of accuracy for heterogeneous
structures due to their inability to enforce physical constraints.

This paper addresses the gap by synergistically combining [advantage
of physics] with [advantage of data-driven], achieving both speed
and accuracy without sacrificing physical fidelity.
```

---

## 1.7 确认你的核心贡献（Contribution）

**贡献陈述金律：**
- 每条贡献都有**量化支撑**（不是形容词）
- 3-5 条为宜
- 用数据说话，不是"提出了一个方法"

**标准格式：**

```
The main contributions of this paper are threefold:
  1) We propose [方法], which achieves [量化性能]. This is the first
     work to [具体创新点].
  2) We develop a [架构/模型] that explicitly captures [特性]
     through [技术].
  3) We demonstrate [效果] on [应用场景], with [量化提升]
     compared with [baseline].
```

---

## 1.8 准备阶段检查清单

在开始写作之前，确认以下全部完成：

- [ ] 已确定目标期刊（IEEE TPEL / TCPMT / TCAD？）
- [ ] 已分析目标期刊读者的典型怀疑点（见 1.5）
- [ ] 已建立"你错了"结构（见 1.2）
- [ ] 已收集 10+ 个目标期刊的价值词汇（见 1.3）
- [ ] 已完成 Related Work 批判性定位（见 1.6）
- [ ] 已确定 3 条量化贡献（见 1.7）
- [ ] 所有实验数据已整理完毕

---

# 第二阶段：起草（按顺序写各部分）

> **写作顺序的原则：** 先写读者最需要看的部分，再写次重要的部分。
> Introduction 是读者第一个看的，但贡献是读者决定是否继续读的理由。
> 因此先明确贡献，再写 Introduction，让整个 Introduction 都指向贡献。

## 2.1 起草顺序

**正确的写作顺序（不是阅读顺序）：**

```
Step 1: 明确贡献陈述（1页以内）
    ↓
Step 2: Figure 设计（围绕贡献组织图表）
    ↓
Step 3: Methods（写你怎么做的，可先有图）
    ↓
Step 4: Results（展示贡献的数据支撑）
    ↓
Step 5: Discussion（解释意义和局限）
    ↓
Step 6: Conclusion（从 Results 自然生长出来）
    ↓
Step 7: Introduction（最后写，因为你知道全部内容后才知道怎么定位）
    ↓
Step 8: Abstract（写完 Introduction 后再写）
```

**为什么 Introduction 最后写？** 因为写完 Methods 和 Results 之后，你才真正知道你的论文对读者意味着什么，才能准确写出"你错了"结构。

---

## 2.2 Abstract 写法

**字数：** 150-200 words（IEEE Transactions 通常 150-200）

**标准结构（4 句）：**

| 句 | 内容 | 时态 |
|----|------|------|
| 第1句 | 背景/问题 | 现在时 |
| 第2句 | 本文方法 | 过去时描述工作 |
| 第3句 | 主要结果（量化） | 过去时 |
| 第4句 | 意义/影响 | 现在时 |

**禁止：** 不要在摘要里引用文献、不要用缩写（除非在该摘要内首次定义）

**模板：**

```
[背景] ...has become a key technology for ... due to [reason].
[问题] However, [challenge] remains unsolved due to [limitation].
[方法] This paper proposes a [method] that [key mechanism], where
[technical detail].
[结果] Experimental results show that the proposed scheme achieves
[metric] of [value], which is [X%] [comparison] the state-of-the-art
under [condition].
[意义] The proposed [method] enables [application] with [benefit],
offering a promising solution for [broader impact].
```

---

## 2.3 Introduction 写法（四段式）

**字数：** 通常 0.8-1.5 pages（双栏格式）

### 第1段：领域背景（3-5 句）

**目的：** 从宽到窄，确立应用重要性

**注意：** 不要用 "In recent years" 这种俗套开场，直接说事实

**模板：**

```
[应用领域] has driven increasing demand for [技术/方法] due to [需求].
[具体场景] presents significant challenges in [具体问题], where
[关键参数] must be carefully managed to ensure [系统性能].

Example:
The exponential growth in 3D-IC power density has driven increasing
demand for fast thermal simulation tools. Accurate temperature
prediction is critical for ensuring reliable operation of multi-core
processors, where peak junction temperatures must be kept below
85°C to maintain performance and longevity.
```

### 第2段：现有方法及其局限性（3-6 句）

**目的：** 指出 gap，建立本文的价值

**模板：**

```
Several approaches have been developed to address this problem.
[X] proposed [方法A] in [年份], which achieves [优点] but suffers
from [局限性]. Similarly, [Y] introduced [方法B], demonstrating
[性能] however still limited by [具体问题].

However, [最关键局限性], which is essential for [应用场景],
has not been effectively addressed by existing methods. This
gap motivates the present work.
```

### 第3段：本文方法与贡献（3-5 句 + 编号列表）

**目的：** 建立"你错了"结构的第三步：给出本文方案

**这是 Introduction 最关键的段落。**

**模板：**

```
To address the aforementioned gap, this paper proposes a [method]
that [核心机制]. Unlike existing approaches, the proposed method
explicitly accounts for [被忽略的因素] through [技术手段].

The main contributions of this paper are threefold:
  1) We propose [方法], which achieves [量化性能]. This is the
     first work to [具体创新].
  2) We develop a [架构] that captures [特性] through [技术],
     enabling [能力].
  3) We demonstrate [效果] on [数据集/应用], achieving [提升]
     compared with [baseline].
```

### 第4段：论文结构 Roadmap（2-3 句）

**模板：**

```
The remainder of this paper is organized as follows. Section II
describes the [principle/formulation] of the proposed method.
Section III presents the [experimental setup and results].
Section IV discusses the [implications and limitations] of
these results. Section V concludes the paper with a summary
and future directions.
```

---

## 2.4 Methods 写法

**原则：**
- 给出足够细节让读者复现
- 解释**为什么**这么做，不只是描述你做了什么
- 变量第一次出现要明确定义

**结构：**

```
A. 数学基础（背景/公式）
   The thermal behavior is governed by the heat equation:
   ρc_p ∂T/∂t = ∇·(k∇T) + Q
   where ρ is the density...

B. 边界条件
   The Robin boundary condition at the package surface:
   -k ∂T/∂n = h(T - T_∞)
   where h is the convective heat transfer coefficient...

C. 本文方法描述（按步骤）
   The proposed framework consists of [X components]:
   Step 1: [description]
   Step 2: [description]
   Step 3: [description]

D. 训练细节（如适用）
   | 内容         | 值         |
   |------------|------------|
   | Optimizer   | Adam       |
   | Learning rate | 1e-3     |
   | Batch size | 32         |
   | Epochs     | 500        |
```

**被动语态为主：** "the converter was operated at...", "the temperature was measured by..."

---

## 2.5 Results 写法

**原则：**
- 先陈述结论，再指向图表
- 每个结论都有数字
- 解释异常值的原因

**结构：**

```
A. 实验设置概述（1-2 句）
   To validate the effectiveness of the proposed method, experiments
   were conducted on [platform/dataset] under [conditions].

B. 主要结果（按指标逐项）
   As shown in Fig. X, the proposed method achieves [metric] of
   [value], which is [X%] higher than [baseline].

   This improvement can be attributed to [physical/mechanism reason]:
   as illustrated in Fig. Y, [specific observation].

C. 对比结果（表格）
   Table II summarizes the performance comparison with
   state-of-the-art methods. The proposed method achieves the
   best performance across all metrics.

D. Ablation Study（如适用）
   To validate each component's contribution, ablation experiments
   were conducted. Removing the [component] results in [degradation],
   confirming its importance for [specific function].
```

**对比表格格式：**

| Method | Peak Error (°C) | MAE (°C) | Runtime (ms) | Speedup |
|--------|-----------------|----------|-------------|---------|
| HotSpot | 4.2 | 2.8 | 120 | 1× |
| U-Net | 2.1 | 1.5 | 8.5 | 14× |
| **Proposed** | **1.3** | **0.9** | **2.1** | **57×** |

最佳值加粗，标注 speedup

---

## 2.6 Discussion 写法

**原则：**
- 解释为什么得到这个结果（physical interpretation）
- 联系已有理论/模型
- 承认局限性
- 指出对工程师的实际意义

**结构：**

```
A. 结果的意义（3-5 句）
   The results demonstrate that [finding]. This behavior can be
   attributed to [physical reason]. Specifically, [mechanism explanation].

B. 与现有理论对照（2-3 句）
   This finding is consistent with the analysis presented in [ref],
   which predicted that [theory prediction]. The observed [phenomenon]
   further confirms [theoretical basis].

C. 局限性承认（1-2 句）
   It should be noted that the present study has some limitations.
   The proposed method was validated on [specific scenario], and its
   performance on [other scenarios] remains to be investigated.

D. 实际意义（1-2 句）
   For practicing engineers, the proposed method offers a practical
   solution for [application], enabling [benefit] without requiring
   [cost/resource].
```

---

## 2.7 Conclusion 写法

**原则：**
- 总结**主要发现**（不是重述方法）
- 贡献重声（用一句简洁的话）
- 承认局限性
- 指出未来方向（自然延伸，不是凑数）
- 不引入新数据或新论证
- 不要用 "In conclusion..." 开头

**模板（应用类论文）：**

```
In summary, this paper addresses [topic] through [核心方法].
The theory for [specific aspect] is based on [principle].
Specific results demonstrate that [quantitative finding 1].
A [device/configuration] achieves [quantitative finding 2],
which represents [X%] improvement over [baseline].

The proposed method enables [capability] for the first time.
We expect this work will serve as an inspiration for [领域]
to [更广阔的意义].
```

**Future Work 的3种模式：**

| 模式 | 位置 | 适用场景 |
|------|------|---------|
| 文内自然延伸 | 结论最后一段 | 应用类论文 |
| 独立小节 | Conclusion 之后 | 综述类论文 |
| Outlook 小节 | Conclusion 之前 | 综述类论文 |

**Future Work 写作原则：** 3-5 条，每条具体，不是泛泛说"more research needed"

---

## 2.8 Figure 设计规范

**总原则：** 每张图必须有明确信息传递，读者看图后应能得出一个具体结论

### 坐标图（波形、效率曲线、Bode）

```
轴标签：Quantity [Unit] 格式，如 "Efficiency [%]"
图例：放在图内右侧或下方，简洁
子图：(a), (b), (c) 在 caption 和图中都要标注
```

### 逻辑图/架构图（最重要的图之一）

```
模块化设计：
- 用方框表示功能模块
- 箭头表示数据/信号流向
- 关键参数标注在箭头旁
- 颜色使用专业（避免过于鲜艳）
- 考虑黑白打印兼容性
```

### 热图/温度分布图

```
温度标尺（colorbar）必须显示
标注最高温度点
有对比时并排放置
```

### Figure Caption 标准格式

```
Fig. X. [动词现在时: shows, illustrates, compares, demonstrates]
[what] under [condition]. [Optional note: Note that ...]
```

**IEEE 要求：** Figure caption 在图表**下方**；Table caption 在表格**上方**

---

## 2.9 LaTeX 格式规范

### 公式格式

```
单公式：
\begin{equation}\label{eq:heat}
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q
\end{equation}

多行公式（对齐）：
\begin{align}
\mathbf{K} \mathbf{T} &= \mathbf{Q} \label{eq:thermal} \\
\text{subject to:} \quad T_i &\geq T_{\text{min}} \label{eq:constraint}
\end{align}
```

**公式后文本延续：** 如果以 "where"/"for"/"with" 开头，加 `\noindent`

### 变量格式

```
斜体：$V_{dc}$, $i_L$, $\omega$
数字和单位不用斜体：10 A, 400 V
运算符不用斜体：sin, cos, d, ∫
```

### 图片浮动体

```
默认：[htbp]（允许 LaTeX 优化）
默认宽度：0.75-0.85\textwidth（不是 0.95）
只有精确位置要求时：[H]
```

---

# 第三阶段：完善（各部分打磨）

> **核心问题：这一段是否建立了对读者的价值？**

## 3.1 各部分完善标准

### Abstract 完善

- [ ] 4 句结构完整（背景→方法→结果→意义）
- [ ] 每句都有量化数据
- [ ] 没有引用文献
- [ ] 没有缩写（除非首次定义）
- [ ] 时态正确（过去时描述工作，现在时描述结论）

### Introduction 完善

- [ ] **第1段无俗套开场**（无 "In recent years..."）
- [ ] **"你错了"结构** 完整：赞扬→承认→挑战→方案
- [ ] **前两段有 10+ 个价值词汇**（however, nonetheless, gap, anomaly, inconsistent...）
- [ ] **贡献列表有量化支撑**（不是形容词）
- [ ] **没有逐一罗列文献**（是批判性对话，不是文献列表）
- [ ] **Roadmap 覆盖所有章节**

### Methods 完善

- [ ] **可复现性**：参数、设置、条件都有具体数值
- [ ] **解释"为什么"**：不只是描述做了什么
- [ ] **变量首次定义**：全文一致
- [ ] **公式编号**：右对齐，连续

### Results 完善

- [ ] **先陈述结论，再指向图表**
- [ ] **每个结论都有数字**
- [ ] **异常值有物理解释**
- [ ] **有 Ablation Study**（ML/热管理论文必须有）
- [ ] **有泛化性测试**（不在训练集上的结果）

### Discussion 完善

- [ ] **有 Physical interpretation**（不只是重复数字）
- [ ] **有局限性承认**（诚实，审稿人欣赏）
- [ ] **有实际意义**（对工程师的意义）
- [ ] **没有引入新数据**

### Conclusion 完善

- [ ] **不是摘要的重复**（结论说"意味着什么"，摘要说"做了什么"）
- [ ] **有量化数据**
- [ ] **有局限性承认**
- [ ] **有 Future Work**（自然延伸）
- [ ] **没有用 "In conclusion..." 开头**

---

## 3.2 避免 AI 痕迹（最常见问题）

**语言层面：**

| AI 痕迹 | 正确做法 |
|---------|---------|
| "In today's rapidly evolving world..." | 直接说事实 |
| "Furthermore"/"Moreover"（每段开头）| 用 however/although/netheless |
| "Importantly,"（每段开头）| 删除 |
| "Firstly, secondly, thirdly" | First, second, third 或数字 |
| "In a nutshell," / "All in all," | In summary, Overall |
| "It is worth mentioning that..." | 删除或改成具体说明 |
| "The experimental results vividly demonstrate..." | Results show that |

**贡献陈述：**

| AI 痕迹 | 正确做法 |
|---------|---------|
| "First work to..." | 谨慎使用，必须有文献证明 |
| "Outperforms all existing methods" | 给出具体对比数据 |
| "Significantly better"（无量化）| "by X%" |
| "Achieves revolutionary results" | 给出具体指标 |

**最高级滥用（无数据支撑）：**

- ❌ "most advanced", "best ever", "unprecedented"
- ✅ 用数据说话：98.7% accuracy, 57× speedup

---

## 3.3 常见语法陷阱

**陷阱1: 主谓不一致**
- ❌ "The parameters shown in Table 1 indicate..."
- ✅ "The parameters shown in Table 1 indicate..." 或 "As shown in Table 1, the parameters indicate..."

**陷阱2: that/which 混用**
- IEEE 严格区分限制性 (that) 和非限制性 (which) 定语从句
- 非限制性定语从句前一定有逗号

**陷阱3: 数字和单位之间缺空格**
- ✅ "10 A", "400 V", "50 kHz"
- ❌ "10A", "400V"

**陷阱4: Figure 缩写使用**
- Fig. 2（正文缩写）
- Figure 2（句首或需要强调时）

---

## 3.4 句式变化

**问题：** 全篇都是长句或都是短句

**解决：** 长句（40-50 词）和短句（15-20 词）交替使用

```
短句：The proposed method achieves 57× speedup.
长句：Compared with the baseline HotSpot simulator, which requires
approximately 120 ms per thermal map generation on a standard
workstation, the proposed approach completes the same computation
in only 2.1 ms while maintaining accuracy within 0.9°C MAE.
```

---

# 第四阶段：修改（投稿前检查）

> **完成第三阶段后，休息一天，再做第四阶段。**
> 新鲜的头脑更容易发现问题。

## 4.1 AI 痕迹终极检查（Humanizer 29 模式）

> 本节基于 Wikipedia "Signs of AI writing" 项目，整合 29 种 AI 生成文本特征。来源：`humanizer` skill

**核心原则：**
- **扫描 AI 模式** → **重写问题段落** → **保留原意** → **增加灵魂**
- AI 文本的最大问题：过于"统计上最安全"，缺乏个性和观点

---

### 语义膨胀类（过度强调意义）

**问题词：** stands as, serves as, is a testament to, vital/significant/crucial/pivotal/key role, underscores, highlights its importance, reflects broader, symbolizing, setting the stage for, marking/shaping, represents a shift, key turning point, evolving landscape, indelible mark

**修复：** 直接陈述事实，不要加"这代表了..."

```
❌ Before: HotSpot has become a cornerstone in thermal modeling, underscoring its pivotal role in modern VLSI design.

✓ After: HotSpot (2006) introduced RC network-based compact thermal modeling for early-stage VLSI design.
```

---

### 媒体/重要性夸张类

**问题词：** independent coverage, widely acclaimed, active social media presence, leading experts

**修复：** 引用具体来源，不要泛泛说"专家表示"

```
❌ Before: Experts believe this approach will reshape the field.

✓ After: In a 2024 IEEE TPS survey, 73% of thermal engineers reported adopting ML-based thermal tools.
```

---

### -ing 形式表面分析

**问题词：** highlighting, underscoring, emphasizing, ensuring, reflecting, symbolizing, contributing to, cultivating, showcasing, encompassing

**修复：** 直接描述，不要用 -ing 形式假装分析

```
❌ Before: The thermal gradient pattern varies significantly, reflecting the non-uniform power dissipation across the die.

✓ After: The thermal gradient varies with power density. Higher power regions show temperature increases of up to 15°C.
```

---

### 宣传性语言

**问题词：** boasts a, vibrant, rich heritage, profound, enhancing its, showcasing, exemplifying, groundbreaking, renowned, breathtaking, stunning, nestled

**修复：** 保持中性学术语气

```
❌ Before: Our method boasts unprecedented accuracy, showcasing remarkable performance improvements.

✓ After: The proposed method achieves 0.32°C MAE, compared with 0.89°C for the baseline.
```

---

### 模糊归因/骑墙表达

**问题词：** Industry reports suggest, Observers have cited, Experts argue, Some critics argue, several sources/publications

**修复：** 具体引用谁，在哪一年，什么结论

```
❌ Before: Industry experts argue that physics-informed loss improves generalization.

✓ After: Raissi et al. (2019) demonstrated that physics constraints reduce OOD error in PINN frameworks.
```

---

### 公式化"挑战与展望"段落

**问题词：** Despite its... faces several challenges..., Despite these challenges, Challenges and Legacy, Future Outlook

**修复：** 说具体问题和具体解决进展

```
❌ Before: Despite its accuracy, the method faces challenges. Future work should explore...

✓ After: The main limitation is computational cost: 45 minutes per full-chip simulation. A recent approximation approach (Chen et al., 2024) reduces this to 12 minutes.
```

---

### 高频 AI 词汇

**问题词（2023年后文本中高频出现）：**
Actually, additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate, pivotal, showcase, tapestry, testament, underscore, valuable, vibrant

**修复：** 用更具体的词替换，或直接删除

```
❌ Before: Additionally, the method demonstrates enhanced performance.

✓ After: The method reduces MAE from 0.89°C to 0.32°C.
```

---

### 系词替换（Copula Avoidance）

**问题词：** serves as, stands as, marks, represents, boasts, features, offers

**修复：** 直接用 is, has, does

```
❌ Before: The framework serves as an efficient solution for thermal simulation.

✓ After: The framework reduces simulation time from hours to milliseconds.
```

---

### 负面并列/尾巴否定

**问题词：** Not only...but..., It's not just about..., no guessing, no wasted motion

**修复：** 直接说正面

```
❌ Before: It's not just about accuracy, it's about efficiency.

✓ After: Accuracy improved 2.7× and inference speed increased 840×.
```

---

### 三点规则滥用

**问题词：** 强制把内容分成三部分（创新、效率、准确性）

**修复：** 自然段落，想分几点分几点

```
❌ Before: Three key advantages: innovation, efficiency, and accuracy.

✓ After: The main advantages are speed and accuracy. Speed comes from the operator learning formulation; accuracy is maintained within 0.5°C MAE.
```

---

### 同义词替换循环（Elegant Variation）

**问题词：** protagonist/character/hero, method/approach/technique/proposed solution 反复切换

**修复：** 保持术语一致

```
❌ Before: The technique was proposed. The method achieves... The approach was validated.

✓ After: DeepOHeat was proposed in 2023. The method achieves R² > 0.99 in-distribution.
```

---

### 假范围（False Ranges）

**问题词：** from X to Y（X和Y不在同一尺度）

```
❌ Before: Results improved from baseline to state-of-the-art.

✓ After: Compared with HotSpot (baseline), the proposed method reduces error by 64%.
```

---

### 被动语态过多

**问题词：** 无主语或隐藏主语的被动句

**修复：** 主动语态+明确主语

```
❌ Before: It can be observed that the temperature increases.

✓ After: The temperature increases by 12°C when power doubles.
```

---

### 破折号滥用

**问题词：** 连续使用 —（EM dash）来制造"有力"效果

**修复：** 大多数用逗号、句号或括号

```
❌ Before: The model achieves 0.32°C MAE—a 2.8× improvement—while reducing inference time by 840×.

✓ After: The model achieves 0.32°C MAE (2.8× better than baseline) and reduces inference time by 840×.
```

---

### 加粗滥用

**修复：** 只在真正需要强调时用，不要每个术语都加粗

```
❌ Before: The **SetFNOModel** architecture combines **Transformer** encoder with **FNO** decoder.

✓ After: The SetFNOModel combines a Transformer encoder with an FNO decoder.
```

---

### 列表标题式垂直列表

**问题词：** 标题+冒号+一句描述的列表

**修复：** 自然段落叙述

```
❌ Before:
- **Accuracy:** Improved by 2.8×
- **Speed:** Increased 840×
- **Generalization:** Extended to 5× more components

✓ After:
Accuracy improved 2.8×, inference speed increased 840×, and generalization extended to 5× more component configurations.
```

---

### 标题中的首字母大写（Title Case）

**修复：** 标题只有首词和专有名词首字母大写

```
❌ Before: ## Strategic Negotiations And Global Partnerships

✓ After: ## Strategic negotiations and global partnerships
```

---

### Emoji 装饰

**修复：** 删除所有 emoji，用文字代替

```
❌ Before: 🚀 Fast | ✅ Accurate | 🔬 Robust

✓ After: Fast, accurate, and robust across OOD configurations.
```

---

### 直引号替换

**修复：** 用 "..." 而不是 "..."

```
❌ Before: The method is "significantly" better.

✓ After: The method achieves 0.32°C MAE, compared with 0.89°C for the baseline.
```

---

### 客服话术（Chatbot Artifacts）

**问题词：** I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., let me know, here is a...

**修复：** 直接进入内容，不要铺垫

```
❌ Before: Here is an overview of our method. I hope this helps!

✓ After: This section describes the proposed method.
```

---

### 知识截止日期免责声明

**问题词：** as of [date], Up to my last training update, While specific details are limited..., based on available information...

**修复：** 不要留免责声明，说确定的事

```
❌ Before: While specific details are limited, the approach seems promising.

✓ After: The approach (proposed in 2023) has been validated on 12 test configurations.
```

---

### 谄媚语气

**问题词：** Great question!, You're absolutely right!, That's an excellent point!

**修复：** 直接回应问题

```
❌ Before: Great question! You're right that this is complex.

✓ After: The thermal coupling between layers is complex due to variable heat paths.
```

---

### 填充短语

| 修复前 | 修复后 |
|--------|--------|
| In order to achieve this goal | To achieve this |
| Due to the fact that | Because |
| At this point in time | Now |
| It is important to note that | [删除] |
| The system has the ability to process | The system can process |
| In the event that you need help | If you need help |

---

### 过度hedging

```
❌ Before: It could potentially possibly be argued that the policy might have some effect on outcomes.

✓ After: The policy may affect outcomes.
```

---

### 通用的正面结尾

**问题词：** The future looks bright, Exciting times lie ahead, a major step in the right direction

**修复：** 说具体未来计划

```
❌ Before: The future looks bright for this technology.

✓ After: Future work will explore adaptive physics loss weighting for dynamic power profiles.
```

---

### 连字符词组滥用

**问题词：** third-party, cross-functional, client-facing, data-driven, decision-making, well-known, high-quality, real-time, long-term, end-to-end

**修复：** 不常见的技术复合词可以保留，但避免所有常见词都加连字符

```
❌ Before: The cross-functional, data-driven approach achieved high-quality, real-time results.

✓ After: The approach achieved real-time accuracy across all test configurations.
```

---

### 说服性权威措辞

**问题词：** The real question is, at its core, in reality, what really matters, fundamentally, the deeper issue, the heart of the matter

**修复：** 直接说问题，不要假装揭示深层真理

```
❌ Before: The real question is whether the method generalizes. At its core, what matters is the physics.

✓ After: The question is whether the method generalizes to unseen component counts. Physics constraints help.
```

---

### 引导性短语（Signposting）

**问题词：** Let's dive in, let's explore, let's break this down, here's what you need to know, now let's look at, without further ado

**修复：** 直接开始说，不要预告

```
❌ Before: Let's dive into the experimental results. Here's what you need to know.

✓ After: Experimental results are summarized in Table III.
```

---

### 碎片化标题

**问题词：** 标题后跟一行简单重复标题内容的句子

**修复：** 删除过渡句

```
❌ Before:
## Performance
Speed matters.
When users hit a slow page, they leave.

✓ After:
## Performance
When users hit a slow page, they leave.
```

---

### 最终 AI 检测两步法

**第一步：** 问自己"这段话哪里看起来像 AI 写的？"

**第二步：** 标记剩余的 AI 特征，然后重写

**最常见的残留问题：**
- 节奏过于整齐（干净的对比、均匀的段落）
- 引用看起来像捏造的占位符
- 结语过于口号式

---

### 过渡词使用频率检查

| 类型 | 可用 | 避免连续使用 |
|------|------|-------------|
| 挑战类 | however, although, nonetheless | — |
| 递进类 | furthermore, moreover, additionally | 连续用3次以上 |
| 并列类 | and, also | 适度使用 |

---

### 填充短语速查表

| 功能 | AI 填充词 | 自然替换 |
|------|----------|---------|
| 引出观点 | It is important to note that | [直接说] |
| 引出原因 | Due to the fact that | Because |
| 引出目的 | In order to | To |
| 引出时间 | At this point in time | Now |
| 引出条件 | In the event that | If |
| 描述能力 | has the ability to | can |
| 结束语 | I hope this helps | [删除] |

---

## 4.2 LaTeX 编译检查

```
□ 编译无错误，无警告
□ PDF 输出正常，无乱码
□ < 和 > 在文本中正确转义为 $<$ 和 $>$
□ ~ 在文本中用 \textasciitilde
□ 公式后延续文本用了 \noindent
□ 图片宽度在 0.75-0.85\textwidth 之间
□ 浮动体大多用 [htbp]，[H] 仅必要时
□ 表格用 booktabs（\toprule, \midrule, \bottomrule）
□ 表格 caption 在表格上方
□ Figure caption 在图表下方
□ widow/orphan penalty 已设置（\widowpenalty=10000, \clubpenalty=10000）
□ 列表间距已压缩（\setlist[itemize]{nosep}）
```

---

## 4.3 内容完整性检查

```
□ 所有实验数据都在论文中呈现
□ 所有图表都有引用（在正文中提到）
□ 所有引用都有对应的参考文献
□ 贡献列表的每条都有量化支撑
□ Ablation Study 完整（ML/热管理论文）
□ 泛化性测试有结果（不在训练集上的数据）
□ 局限性已承认（审稿人欣赏诚实）
□ Future Work 具体（不是泛泛的"more research needed"）
□ 没有在结论中引入新数据
```

---

## 4.4 格式检查（IEEE Transactions）

```
□ Title 简洁（10-15 词），无缩写（公认除外）
□ Abstract 150-200 词，无文献引用
□ Keywords 4-6 个
□ Introduction 0.8-1.5 页
□ 参考文献按正文引用顺序排列
□ 参考文献格式统一（IEEEtran）
□ 所有缩写首次出现有定义
□ 变量名全文统一
□ 公式编号连续
□ Figure/Table 编号连续
```

---

## 4.5 读者模拟检查（最重要的检查）

**找一位同领域的同学/同事，让他们读你的 Introduction 和 Abstract，然后问：**

1. "这篇论文解决了什么问题？" —— 回答不出来 → Introduction 价值建立不足
2. "它比现有方法好在哪里？" —— 回答不出来 → 贡献陈述不够量化
3. "你有什么疑问？" —— 有疑问 → 你的 Introduction 没有回答读者可能的怀疑

---

## 4.6 投稿前最终检查清单

**40+ 项检查，每项逐一确认：**

### 语言层面
- [ ] 无俗套开场
- [ ] 无每段开头的机械过渡词
- [ ] 句式有长短变化
- [ ] 无最高级滥用（无数据支撑时）

### 事实层面
- [ ] 所有文献引用真实（作者、年份、结论已核实）
- [ ] 所有数字有计算依据
- [ ] 无常识性错误（公式、物理原理）

### 逻辑层面
- [ ] 每段话之间有自然的逻辑连接
- [ ] 推理过程完整，没有跳跃
- [ ] 因果关系正确，不是把相关性当因果

### 格式层面
- [ ] 缩写第一次出现有定义
- [ ] 变量名全文统一
- [ ] 公式编号连续
- [ ] 参考文献格式统一

### 内容层面
- [ ] 贡献陈述有量化支撑
- [ ] 方法描述有足够细节（可复现）
- [ ] 结论不是摘要的重复
- [ ] 有客观承认局限性

### LaTeX 层面
- [ ] 图片宽度在 0.75-0.85\textwidth
- [ ] 浮动体大多用 [htbp]
- [ ] < 和 > 在文本中正确转义
- [ ] 公式后延续文本用了 \noindent
- [ ] 有 widow/orphan penalty
- [ ] 列表间距已压缩

---

# 附录

## 附录 A：IEEE 论文标准结构

```
I. INTRODUCTION                    （引言）
II. [METHOD/THEORY/ANALYSIS]      （方法/理论，按研究内容命名）
III. [EXPERIMENT/RESULTS]         （实验/结果）
IV. [DISCUSSION]                  （讨论，可选）
V. CONCLUSION                      （结论）
ACKNOWLEDGMENT                     （致谢）
REFERENCES                         （参考文献）
APPENDIX                           （附录，可选）
BIOGRAPHIES                        （作者简介，仅 Trans 期刊需要）
```

---

## 附录 B：热管理/ML 论文专用词汇

| 中文 | 英文 | 注意事项 |
|------|------|---------|
| 热阻 | thermal resistance (R_th) | 标注类型：R_jc, R_jh |
| 热容 | thermal capacitance (C_th) | |
| 热阻抗 | thermal impedance (Z_th) | transient 时用 |
| 结温 | junction temperature (T_j) | |
| 相变材料 | Phase Change Material (PCM) | 首次全写 |
| 边界条件 | boundary conditions | 区分 Robin/Neumann/Dirichlet |
| 降阶模型 | Model Order Reduction (MOR) | |

**ML/热管理论文特有 baseline：**
- HotSpot（最常用热建模 baseline）
- FVM/FEM tools（COMSOL, ANSYS IcePak）
- U-Net, ResNet（深度学习 baseline）
- 3D-ICE, MatEx（快速仿真 baseline）

---

## 附录 C：常用句式速查

### 介绍问题/背景
```
[Topic] has become increasingly important in [application] due to [reason].
The growing demand for [X] has motivated extensive research on [Y].
```

### 指出差距/问题
```
However, [existing approaches] still suffer from [limitation].
Despite significant advances in [area], [problem] remains a challenging issue.
A key remaining challenge is [specific problem], which has not been fully addressed.
```

### 提出本文贡献
```
To address this issue, this paper proposes [approach], which [key benefit].
Different from existing methods, the proposed [scheme] [specific advantage].
```

### 描述实验/方法
```
The [equipment] was set to [condition].
The input voltage was varied from X to Y while [parameter] was kept constant.
Measurements were performed at [condition] unless otherwise specified.
```

### 描述结果
```
As shown in Fig. X, the [measured quantity] increases/decreases with [variable].
The efficiency reaches a maximum of X% at [operating point].
Compared with [baseline], the proposed method achieves [improvement].
```

### 讨论含义
```
This behavior can be attributed to [physical reason].
This result indicates that [interpretation].
This finding is consistent with the analysis presented in Section X.
```

---

## 附录 D：常见短语搭配

| 功能 | 常用表达 |
|------|---------|
| 引用图表 | "as shown in Fig. X", "Fig. X illustrates", "it can be seen from Fig. X" |
| 引用公式 | "defined as (X)", "expressed as (X)", "given by (X)" |
| 条件状语 | "when X is at Y", "if X exceeds Y", "under [condition]" |
| 对比 | "compared with/against", "in contrast", "unlike [X]" |
| 因果 | "as a result", "consequently", "this leads to", "due to" |
| 转折 | "however", "nevertheless", "although", "despite" |
| 强调 | "in particular", "specifically", "notably", "it should be noted" |

---

## 附录 E：论文长度参考（IEEE Transactions，8-10 页双栏）

| Section | 参考长度 |
|---------|---------|
| Introduction | 1-1.5 pages |
| Methodology | 3-4 pages |
| Results | 2-3 pages |
| Discussion | 1-1.5 pages |
| Conclusion | 0.5 page |

---

## 附录 F：推荐参考论文（写作和作图模板）

### 写作质量最高
- **"Control of Grid-Forming VSCs: A Perspective of Adaptive Fast-Slow Internal Voltage Source"** — Introduction 逻辑最清晰，Discussion 有深度
- **"Parameter Estimation of Power Electronic Converters With Physics-Informed Machine Learning"** — 写作规范，方法描述详尽

### 图质量最高
- **"Vertical Stacked LEGO-PoL CPU Voltage Regulator"** — 逻辑图精美，模块化清晰
- **"Liquid Metal Fluidic Connection..."** — 封装细节图精致
- **"First Characterization of Si IGBT- SiC MOSFET- and GaN HEMT..."** — 波形图规范

### 控制环图参考
- **"A Family of Symmetrical Integrated Synchronizations for Grid-Following and Grid-Forming Inverters"** — 同步控制环图
- **"Overcurrent Limiting in Grid-Forming Inverters"** — 限流控制环图

---

# 更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-04-01 | 初始版本：基于40篇TPEL论文阅读总结 |
| 2026-04-01 | 全面重构为四阶段写作体系（准备→起草→完善→修改）|
| | 整合：57篇thermal journals分析、芝加哥大学写作方法论、Cambridge Research写作指南、IEEE官方结构指南 |
| | 新增：第一阶段（读者思维、"你错了"结构、社区密码）|
| | 新增：第二阶段（8步写作顺序、各部分标准写法）|
| 2026-04-02 | 整合 humanizer skill（29种AI写作模式）到第四阶段，替换原有的简单检查列表 |
| | 新增：详细的AI特征分类（语义膨胀、高频词汇、被动语态、填充短语等）|
| | 新增：每种模式的修复前后对比示例 |
| | 新增：最终AI检测两步法 + 残留问题速查 |
| | 新增：第四阶段（LaTeX检查、内容完整性、格式检查）|
| | 新增：附录A-F（结构模板、专用词汇、句式速查、参考论文）|
