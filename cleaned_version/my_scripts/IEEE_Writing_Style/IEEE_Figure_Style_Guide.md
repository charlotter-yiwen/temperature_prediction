# IEEE TPEL 图风格规范 — 重点：逻辑图与系统框图

> 基于 40 篇 IEEE TPEL 论文阅读总结
> 与 IEEE_TPEL_Writing_Style_Guide.md 配套使用

---

## 一、好的逻辑图的核心标准

### 1.1 必须传达的信息（自明性）
- 看图的人**不需要读正文**就能理解系统的核心结构
- 每条信号线有明确指向
- 每个模块/方框有清晰名称

### 1.2 布线原则
- **信号流从左到右**（最常见，符合阅读习惯）
- 或**从上到下**（适用于多层控制环）
- 避免线与线交叉（太多交叉说明布局有问题）
- 线条**粗细一致**（power 线可略粗，signal 线略细）

### 1.3 颜色使用
- **IEEE 鼓励黑白兼容** — 好的图在黑白打印下依然清晰
- 如果用颜色，同一类型的线/模块用**同一色系**
- 避免过多颜色（一般不超过 4-5 种）
- 传递函数框用**浅色填充**（如浅蓝、浅灰）

### 1.4 字体
- 模块内文字：**清晰可读**，字号不小于 8pt
- 轴标签：**12pt 左右**
- 图例：**10pt 左右**
- 同一图中**字体风格统一**

---

## 二、系统框图标准结构

### 2.1 控制环通用模板

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐
│ Reference│────▶│ Controller│────▶│ PWM/Mod │────▶│ Power    │
│ (v*, i*)│     │ (Gc)     │     │         │     │ Stage    │
└─────────┘     └──────────┘     └─────────┘     └──────────┘
                    ▲                                      │
                    │                                      ▼
              ┌──────────┐                         ┌──────────┐
              │ Feedback  │◀────────────────────────│ Output   │
              │ (H)       │                         │ (Vo, Io) │
              └──────────┘                         └──────────┘
```

**各部分规范：**

| 模块 | 内容 | 常用标注 |
|------|------|---------|
| Reference | 参考值 | $v^*$, $i^*$, $P^*$, $Q^*$ |
| Controller | 控制器 | $G_c$, PI, PR, Deadbeat |
| Modulation | 调制 | PWM, SVM, DPWM |
| Power Stage | 功率级 | $G_p(s)$, Converter |
| Feedback | 反馈 | $H$, Sensor, Sampler |
| Output | 输出 | $v_o$, $i_o$, $P_o$ |

### 2.2 采样点表示
- 用 **圆点 ●** 表示采样点
- 用 **×** 表示比较器（有时）

### 2.3 信号类型区分
- 实线：主信号流
- 虚线：反馈信号
- 点划线： enable/disable 或 mode switch

---

## 三、论文中观察到的具体图例

### 3.1 控制环图 — 优秀范例

**来源1：Control of Grid-Forming VSCs**
- 特点：Fast-Slow 内电压源架构清晰，两个时间尺度用不同填充色区分
- 借鉴点：双时间尺度框架用颜色/布局区分，是审稿人喜欢的创新展示方式

**来源2：Virtual-Impedance-Based Control**
- 特点：清晰区分 voltage-source 和 current-source 两类 converter
- 借鉴点：同一张图对比两种 converter 时，用统一框架但标注差异

**来源3：Overcurrent Limiting in Grid-Forming Inverters**
- 特点：限流环嵌入主控环的层次关系表达清楚
- 借鉴点：限流 logic 用虚线框包围，视觉上表示 conditional activation

### 3.2 电路拓扑图 — 优秀范例

**来源：Bidirectional Onboard Chargers for EV**
- 特点：多端口 power flow 方向用**箭头标注**，清晰展示能量流向
- 借鉴点：功率流向图必须标注方向，否则读者困惑

**来源：98.3% Efficient Multiport System**
- 特点：模块化程度高，每个模块用数字标注，右侧列出模块清单
- 借鉴点：复杂系统用编号代替框内文字，右侧 legend 说明

### 3.3 效率曲线图 — 规范格式

**标准布局：**
```
Y轴: Efficiency [%] 或 Power Loss [W]
X轴: Load Power [W] 或 Output Current [A]
曲线: 多个条件（不同 color/marker）对比
图例: 放在图内右下角或图下方
```

**优秀范例：**
- "Analysis and Optimization of Switched-Capacitor DC-DC Converters" — 效率曲线对比清晰
- "Bidirectional Onboard Chargers" — 多条件效率 map 规范

### 3.4 Bode Plot 规范

**标准布局：**
```
两个子图上下排列：
(a) Magnitude [dB] vs Frequency [Hz]
(b) Phase [deg] vs Frequency [Hz]
```

**关键标注：**
- 穿越频率 (crossover frequency) 用**垂直虚线**标注
- 相位/增益裕度在图内直接标数值
- 重要极点/零点用**箭头**指示

---

## 四、自省：哪些图可以写得更好

### 4.1 常见图的问题

**问题1：模块内文字过小**
- 审稿人/读者第一眼就看不清楚，印象分大减
- **改进**：文字至少 10pt，或把文字放到模块旁边用引线连接

**问题2：颜色过杂**
- 超过 5 种颜色的图很难读
- **改进**：用 line style (实线/虚线/点线) + color 组合区分，最多 3-4 颜色

**问题3：图例文字过长**
- 图例占满半个图
- **改进**：图例最多 2-3 个关键词，详细说明放 caption

**问题4：坐标轴标签不规范**
- 用 "Effi." 代替 "Efficiency"，用 "freq" 代替 "Frequency"
- **改进**：坐标轴标签必须用完整单词

**问题5： Caption 太短**
- Caption 只写 "Efficiency curve" — 毫无信息量
- **改进**：Caption 必须包含：什么条件、什么对比、什么结论

### 4.2 可以进一步提升的地方

- **多级标题图**：如果系统有多层控制环，每一层的图应该风格统一但有层次区分
- **动画/动态图**：电力电子论文中用动态 GIF 展示启动过程正在流行
- **3D 图**：效率 map 等用 3D surface plot 是趋势，但要注意 projection 是否清晰

---

## 五、生成图的代码规范

### 5.1 Python/Matplotlib 关键设置

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# 坐标轴
ax.set_xlabel('Output Power [W]', fontsize=12)
ax.set_ylabel('Efficiency [%]', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# 网格（一般不用，电力电子图一般不用网格）
# ax.grid(True, linestyle='--', alpha=0.5)

# 图例
ax.legend(loc='best', fontsize=10, frameon=False)

# 边距
plt.tight_layout()
```

### 5.2 电路图工具推荐
- **Visio**：最常用
- **Lucidchart**：在线协作
- **Draw.io**：免费开源
- **Python (schemdraw)**：代码绘制电路图，适合需要版本控制的场景

### 5.3 逻辑图/框图工具推荐
- **Visio**
- **PowerPoint**（快速简单）
- **Draw.io**
- **TikZ/PGF**（LaTeX 代码绘制，IEEE 论文常用，质量最高）
- **Matplotlib + annotate**（代码控制，适合程序化生成）

### 5.4 导出设置
- **矢量格式优先**：PDF、SVG、EPS（可无损缩放）
- **位图格式**：PNG（如果必须），至少 300 dpi
- **颜色空间**：RGB（期刊印刷会用 CMYK，不需要自己转换）

---

## 六、热管理论文图表专题（基于57篇 thermal journals 分析）

### 6.1 热图（Thermal Map）规范

热图是热管理论文的核心图表，必须规范：

**标准热图要素：**
```
┌─────────────────────────────────────┐
│  (a) Temperature distribution       │
│                                      │
│    [热图图像]                        │
│                                      │
│  Colorbar: [40°C----80°C----120°C]  │
│                                      │
│  注：颜色从蓝(冷)到红(热)渐变         │
└─────────────────────────────────────┘
Caption: Fig. X. Temperature distribution of (a) proposed method and (b) HotSpot
under the same power map. Colorbar indicates temperature in °C. The proposed
method achieves MAE of 0.9°C compared with ground truth FEA simulation.
```

**热图配色标准：**
- 冷到热：Blue → Cyan → Green → Yellow → Red（或 jet / turbo colormap）
- **必须包含 colorbar**，标注温度单位 (°C)
- 同一论文中**多张热图用相同色标范围**（方便对比）
- 重要温度点用 **●** 标注（如最高温度点）

**热图对比布局：**
```
Fig. X. Temperature distributions comparison under [condition]:
(a) Ground truth (FEA), (b) HotSpot, (c) U-Net, (d) Proposed method.
All methods use the same color scale [30°C - 120°C]. The proposed method
achieves the best visual agreement with ground truth.
```

### 6.2 ML 架构图规范

ML for thermal 论文的核心图表是网络架构图：

**标准架构图要素：**
```
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
│ Input:   │     │ Encoder   │     │ Processor│     │ Decoder   │
│ P ∈ ℝ^{H×W}│ ──▶│ (Conv)   │ ──▶│ (FNO/   │ ──▶│ (Deconv) │
│          │     │           │     │  Transformer)│   │           │
└──────────┘     └───────────┘     └──────────┘     └───────────┘
                                                   Output:
                                                   T ∈ ℝ^{H×W}
```

**关键要求：**
- 每个模块用**圆角矩形**（rounded rectangle）
- 模块内标注**名称 + tensor 形状**（如 P ∈ ℝ^{H×W}）
- 模块间用**箭头**表示数据流（标注箭头方向）
- 相同类型模块用**相同颜色**
- **颜色编码**说明：蓝色=输入、绿色=处理、橙色=输出

**ML 架构图画法推荐工具：**
1. **TikZ/PGF**（LaTeX，IEEE 最常用，质量最高）
2. **Matplotlib + annotate**（Python 代码，可版本控制）
3. **PowerPoint**（快速草图）
4. **Draw.io**（免费，可导出 SVG）

### 6.3 交叉验证图（Cross-validation / Error Distribution）

用于展示预测精度，标准格式：

**Error Heatmap：**
```
Fig. X. Spatial error distribution (predicted - ground truth):
(a) HotSpot, (b) U-Net, (c) Proposed. The proposed method shows
uniformly low error (< 2°C) across the chip, while HotSpot exhibits
localized high error (> 5°C) near hotspots.
```

**Parity Plot（预测 vs 真实）：**
```
Fig. X. Parity plot of predicted vs. ground truth temperature.
Each point represents a spatial location. R² = 0.998. MAE = 0.9°C.
RMSE = 1.2°C. The proposed method shows excellent agreement across
the entire temperature range [40°C - 110°C].
```

### 6.4 效率/性能柱状图

**标准柱状图格式：**
```
Fig. X. Runtime comparison with state-of-the-art thermal simulation
methods. The proposed method achieves 57× speedup over HotSpot and
320× speedup over FEA, while maintaining comparable accuracy (MAE < 1°C).
```

- **柱状图**：用不同颜色区分方法，加上数值标签
- **误差线**：有统计时要加 error bar
- **对数刻度**：runtime 差异大时用 log scale

### 6.5 热阻网络图（Cauer/ Foster Network）

热管理特有图表：

**标准热阻网络图：**
```
        Q_j
         ●──────R_jc──────●──────R_cs──────●──────R_sa──────●
                           │                │
                          C_jc             C_cs
                           │                │
                          GND              GND

Caption: Fig. X. Cauer-type thermal network model of the power module.
R_jc, R_cs, and R_sa represent junction-to-case, case-to-spreader,
and spreader-to-ambient thermal resistances, respectively.
C_jc and C_cs are the corresponding thermal capacitances.
```

### 6.6 封装结构剖面图（Cross-sectional View）

**标准剖面图要素：**
- 所有层用**不同灰度/颜色**区分
- 每层标注**材料名称**（Si, Cu, TIM, PCB 等）
- 标注**关键尺寸**（厚度、宽度）
- 热流方向用**箭头**表示

### 6.7 Transient 曲线图

**标准 transient 图：**
```
Fig. X. Transient junction temperature response to step load change:
(a) from 50W to 100W at t=0, (b) zoomed view of first 100ms.
The proposed method (red) shows excellent agreement with experimental
measurements (black dashed). Steady-state error < 0.5°C.
```

- **双 y 轴**：温度 + 时间（或功率）
- **关键时间点**标注数值
- **稳态值**水平虚线标注

### 6.8 遗传算法/优化流程图

**NSGA-II / 优化算法流程图：**
```
┌──────────────┐
│ DOE: Latin  │
│ Hypercube   │  (N samples)
└──────┬───────┘
       ▼
┌──────────────┐
│ FEA          │  (evaluate all samples)
│ Simulation   │
└──────┬───────┘
       ▼
┌──────────────┐
│ Kriging      │  (build surrogate model)
│ Surrogate    │
└──────┬───────┘
       ▼
┌──────────────┐
│ NSGA-II      │  (multi-objective optimization)
│ Optimization │
└──────┬───────┘
       ▼
┌──────────────┐
│ Pareto Front │──▶ Optimal Design
│ Solutions    │
└──────────────┘
```

### 6.9 论文图表常见错误检查清单

**发布前必查：**
- [ ] Colorbar 有标签和单位
- [ ] 子图有 (a), (b), (c) 标注
- [ ] 坐标轴有完整标签（Quantity [Unit]）
- [ ] 图例清晰可读（≥8pt）
- [ ] Caption 包含"什么 + 条件 + 结论"
- [ ] 矢量格式（PDF/EPS）优先
- [ ] 同一论文风格统一（字体、线条粗细）

### 6.10 图表生成代码模板（Python/Matplotlib）

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 热图绘制
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(temperature_map, cmap='turbo', vmin=30, vmax=120)
cbar = fig.colorbar(im, ax=ax, label='Temperature [°C]')
ax.set_xlabel('X [mm]', fontsize=12)
ax.set_ylabel('Y [mm]', fontsize=12)
ax.set_title('(a) Temperature Distribution', fontsize=12)

# 设置 tick 清晰
ax.tick_params(axis='both', labelsize=10)

# 图例放在合适位置
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('fig_thermal_map.pdf', dpi=300, bbox_inches='tight')
```

---

## 七、持续更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-04-01 | 初始版本：基于40篇TPEL论文阅读，重点整理逻辑图规范 |
| 2026-04-01 | 全面更新：新增第六章热管理论文图表专题，基于57篇thermal journals论文 |
| | 新增：热图规范、ML架构图、交叉验证图、热阻网络、封装剖面图、transient曲线、优化流程图 |
| | 新增：图表生成代码模板、错误检查清单 |
| | 来源：DeepOHeat, ARO, PINN, GAN, Point Cloud Transformer, PCM, 随机求解器等57篇论文 |
