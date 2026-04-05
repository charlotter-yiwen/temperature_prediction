# IEEE TPEL 优秀图例参考清单

> 基于 40 篇 IEEE TPEL 论文阅读，标注值得参考的图例
> 来源文件在：`literature review/IEEE Transaction on power electronics/`

---

## 一、逻辑图 / 系统框图 / 控制环图

### 1.1 控制环架构图

**推荐论文：**
- `Control of Grid-Forming VSCs A Perspective of Adaptive Fast-Slow Internal Voltage Source.pdf`
  - 亮点：双时间尺度 (fast-slow) 架构用**颜色/填充区分**，清晰表达两个控制 loop 的层次关系
  - 值得学：内电压源 adaptive mechanism 的反馈环画法

- `Virtual-Impedance-Based Control for Voltage-Source and Current-Source Converters.pdf`
  - 亮点：同一张图清楚对比 VSG 和 CSI 两种 converter 的控制差异
  - 值得学：virtual impedance 环如何嵌入主控环的可视化方法

- `Overcurrent Limiting in Grid-Forming Inverters A Comprehensive Review and Discussion.pdf`
  - 亮点：限流 logic 用**虚线框**包围，表示 conditional activation
  - 值得学：多个约束条件叠加时如何组织图面

- `Cross-Forming Control and Fault Current Limiting for Grid-Forming Inverters.pdf`
  - 亮点：Fault current limiting 环的层次关系清晰
  - 值得学：故障工况下控制模式切换的图示

### 1.2 系统整体架构图

**推荐论文：**
- `98.3- Efficient Multiport System With Multidirectional Power Flow to Integrate Battery Storage-Renewables With the Grid for EV Charging.pdf`
  - 亮点：多端口能量流向用**箭头**标注，EV 充电、储能、并网三向功率流一目了然
  - 值得学：多端口系统模块编号 + 右侧 legend 的布局

- `A Cascaded Hybrid Synchronization Control for Grid-Connected Inverters.pdf`
  - 亮点：同步控制架构分层清楚
  - 值得学：Phase-Locked Loop (PLL) 嵌入主控环的画法

- `Bidirectional Onboard Chargers for Electric Vehicles State-of-the-Art and Future Trends.pdf`
  - 亮点：OBC 系统图用 power flow 箭头 + 模块化布局
  - 值得学：汽车电子系统图的规范画法

---

## 二、电路拓扑图

**推荐论文：**
- `A Novel Current-Fed Dual-Active-Bridge DC-DC Converter for Ultra-Wide Output Voltage Range Electric Vehicle Battery Charger.pdf`
  - 亮点：current-fed DAB 拓扑清晰，开关状态标注完善
  - 值得学：宽输出电压范围的 converter 拓扑如何简化表达

- `Overview of Dual-Active-Bridge Isolated Bidirectional DC-DC Converter for High-Frequency-Link Power-Conversion System.pdf`
  - 亮点：高频链路 converter 的模块划分清晰
  - 值得学：DAB 拓扑的表达方式

- `Integrated Inductor-Transformers for High-Frequency Converters An Overview.pdf`
  - 亮点：平面变压器结构图精细
  - 值得学：封装/磁性元件图的绘制规范

---

## 三、波形图（Oscilloscope Traces）

**推荐论文：**
- `First Characterization of Si IGBT- SiC MOSFET- and GaN HEMT at Deep Cryogenic Temperatures Down to 10 Millikelvins.pdf`
  - 亮点：深低温下器件特性波形，标注清晰（最高温度点、数值标注）
  - 值得学：极端工况下波形图的标注方式

- `A Family of Symmetrical Integrated Synchronizations for Grid-Following and Grid-Forming Inverters.pdf`
  - 亮点：同步过程波形（锁相、切换过程）清晰标注关键时间点
  - 值得学：动态过程波形的关键帧标注

- `Low-Voltage Ride-Through Algorithm for Grid-Forming Converters.pdf`
  - 亮点：LVRT 工况下的波形图，对称性破缺和恢复过程标注清楚
  - 值得学：故障穿越波形的时间轴对齐

---

## 四、效率 / 功率曲线图

**推荐论文：**
- `Analysis and Optimization of Switched-Capacitor DC-DC Converters.pdf`
  - 亮点：效率曲线多条件对比，不同 duty ratio 用不同 marker 区分
  - 值得学：散点图 + 拟合曲线叠加的画法

- `98.3% Efficient Multiport System...`（同一篇）
  - 亮点：效率 map (heatmap) + 效率曲线组合展示
  - 值得学：效率的 2D heatmap + 1D 曲线结合

- `Parallel Connection of Silicon Carbide MOSFETs-Challenges- Mechanism- and Solutions.pdf`
  - 亮点：器件并联均流特性对比图规范
  - 值得学：多器件并联均流测试数据的展示方式

---

## 五、Bode Plot（频率响应）

**推荐论文：**
- `Impedance Modeling and Analysis of Grid-Connected Voltage-Source Converters.pdf`
  - 亮点：阻抗 Bode plot + Nyquist plot 组合，子图标注清楚
  - 值得学：阻抗分析图的规范格式

- `Impedance-Based Stability Criterion for Grid-Connected Inverters.pdf`
  - 亮点：稳定性判据用 Bode plot + Nyquist 组合，关键相位/增益裕度直接标在图上
  - 值得学：相位裕度、增益裕度数值直接标在图内

- `Unified Impedance Model of Grid-Connected Voltage-Source Converters.pdf`
  - 亮点：多个 converter 阻抗对比，Bode plot 叠加清晰
  - 值得学：多条件阻抗对比的画法

---

## 六、热分析 / 封装图

**推荐论文：**
- `Liquid Metal Fluidic Connection and Floating Die Structure for Ultralow Thermomechanical Stress of SiC Power Electronics Packaging.pdf`
  - 亮点：热成像图 + 封装结构图组合，colorbar 规范
  - 值得学：封装图的结构标注方式

- `Repairable- Recyclable- and Reliable Power Electronics Using Liquid Metal Interconnection.pdf`
  - 亮点：可修复 power electronics 的结构爆炸图
  - 值得学：复杂封装的分层展示方式

- `In-Depth Review of Planar Transformer Modeling and Optimization Challenges and Perspectives.pdf`
  - 亮点：平面变压器结构细节图
  - 值得学：磁性元件图的绘制

---

## 七、流程图 / 算法框图

**推荐论文：**
- `Parameter Estimation of Power Electronic Converters With Physics-Informed Machine Learning.pdf`
  - 亮点：PINN 训练流程图，清晰展示 forward pass + physics residual + 反向传播
  - 值得学：AI + Power Electronics 结合的流程图画法

- `Estimating Electric Motor Temperatures With Deep Residual Machine Learning.pdf`
  - 亮点：Deep residual learning 框架图
  - 值得学：神经网络架构图的简化画法（不需要画所有层）

- `How MagNet Machine Learning Framework for Modeling Power Magnetic Material Characteristics.pdf`
  - 亮点：ML framework 全流程图，从材料测试到模型训练到部署
  - 值得学：复杂 ML pipeline 的模块化图示

---

## 八、数据对比表

**推荐论文：**
- `Parallel Connection of Silicon Carbide MOSFETs-Challenges- Mechanism- and Solutions.pdf`
  - 亮点：器件并联方案对比表，列出各项参数和性能指标
  - 值得学：多方案对比表的排版规范

- `Switched-Capacitor Multilevel Inverters A Comprehensive Review.pdf`
  - 亮点：大表格但排版清晰，分成多个子表按拓扑分类
  - 值得学：巨型对比表如何组织

---

## 九、特别说明：逻辑图核心要素检查清单

画逻辑图/控制环图时，确保包含：

- [ ] 输入参考值（$v^*$、$i^*$等）
- [ ] 控制器方框（标注传递函数或类型：PI/PR/Deadbeat）
- [ ] 调制模块（PWM/SVM/DPWM）
- [ ] 功率级方框
- [ ] 输出反馈路径（虚线）
- [ ] 采样点（●）
- [ ] 比较器节点（如有）
- [ ] 信号流箭头方向
- [ ] 各模块名称

Caption 格式：
```
Fig. X. [What the figure shows]. [Key characteristics or conditions]. 
[Source: this work /仿真 /实验].
```

---

## 十、热管理论文优秀图例清单（基于57篇 thermal journals）

### 10.1 ML 架构图 — 推荐论文

**论文：DeepOHeat-v1 / DeepOHeat**
- 亮点：Operator Learning 框架图清晰，左编码器→物理约束→右解码器，颜色编码一致
- 值得学：损失函数分叉显示（data loss + physics loss 两路）

**论文：A Geometry-Material Aware Point Cloud Transformer for Large-scale Unstructured Thermal Analysis in 2.5D ICs**
- 亮点：双分支架构（Geometry encoder + Material encoder），fusion 模块清晰
- 值得学：Point cloud 可视化 + 变压器架构的组合方式

**论文：Real-Time 3-D Thermal Simulation of Advanced Packages via Generative Adversarial Networks**
- 亮点：GAN 架构图精美，Generator-Discriminator 分支清晰，skip connection 标注
- 值得学：3D 体积热图渲染（多 slice 组合）+ 架构图结合

**论文：Advanced Spatial Temperature Monitoring via Fourier Neural Operator**
- 亮点：FNO 特有的频域分支架构，encoder-decoder 清晰
- 值得学：频域→空域 dual-branch 设计可视化

**论文：Real-time thermal map estimation for AMD multi-core CPUs using transformer**
- 亮点：Transformer 架构图展示 self-attention 机制，CPU floorplan 叠加热图
- 值得学：Attention weight 可视化热图 + 架构图的组合

### 10.2 热图（Temperature Distribution） — 推荐论文

**论文：DeepOHeat-v1 / DeepOHeat**
- 亮点：Ground truth vs. 预测热图并排，共享 colorbar，统一色标
- 值得学：side-by-side 对比 + 误差分布 inset

**论文：Full-chip thermal map estimation for commercial multi-core CPUs with GAN**
- 亮点：CPU floorplan 上叠加热图，colorbar 规范，有 contour line
- 值得学：Floorplan-guided 热图叠加方式

**论文：FaStTherm — Fast and Stable Full-Chip Transient Thermal Predictor**
- 亮点：热图 + 误差分布图组合，transient 曲线叠加
- 值得学：Transient thermal 结果展示方式

**论文：Point Cloud Transformer for 2.5D ICs**
- 亮点：CAD 几何叠加热图，多层结构（4-layer, 8-layer）分别展示
- 值得学：Unstructured 几何的热图展示

### 10.3 热阻网络图（Cauer Network） — 推荐论文

**论文：3D-ICE 系列**
- 亮点：Cauer network RC 梯级清晰，节点标注对应物理层
- 值得学：2D 层间热阻网络的规范画法

**论文：Transient Thermal Characterization of Power Module with PCM**
- 亮点：红外图像 + 热阻网络组合，有效热阻概念图
- 值得学：实验与模型结合的可视化

**论文：Uncertainty Analysis of Interface Thermal Resistance**
- 亮点：Monte Carlo 概率分布图 + 热阻网络组合
- 值得学：不确定性分析的可视化

### 10.4 封装结构剖面图 — 推荐论文

**论文：3D-ICE 3.0 — Efficient nonlinear MPSoC thermal simulation**
- 亮点：多层 3D IC 剖面，液冷通道可视化，heat sink 结构标注
- 值得学：热流方向箭头 + 层间结构标注

**论文：NSGA-II Optimized Manifold Microchannel Heat Sink**
- 亮点：微通道散热器详细剖面，歧管结构标注，**质量极高**
- 值得学：复杂散热器工程图绘制标准

**论文：Multidisciplinary Design Optimization of LCCC Package**
- 亮点：陶瓷封装多层结构，CAD 风格渲染，FEA mesh 可视化
- 值得学：封装工程图 + 优化结果结合

**论文：Compact Transient Thermal Model for 3D ICs with Liquid Cooling**
- 亮点：液冷微通道结构，多视角（top/side/isometric）
- 值得学：3D 结构多视角展示

### 10.5 transient 曲线图 — 推荐论文

**论文：MatEx — Efficient Transient and Peak Temperature Computation**
- 亮点：温度 vs. 时间曲线，transient 响应清晰，有稳态标注
- 值得学：峰值温度计算结果的展示

**论文：FaStTherm — Fast and Stable Full-Chip Transient Thermal Predictor**
- 亮点：多功率步变下 transient 曲线，误差带显示
- 值得学：非稳态热仿真的标准展示

**论文：SiC MOSFET Temperature-Dependent Analytical Transient Model**
- 亮点：开关瞬态温度轨迹，开关损耗叠加
- 值得学：功率器件开关瞬态热分析

### 10.6 优化/Pareto 前沿图 — 推荐论文

**论文：NSGA-II Optimized Microchannel Heat Sink**
- 亮点：Pareto 前沿图，热阻 vs. 压降权衡一目了然
- 值得学：多目标优化结果展示

**论文：Multidisciplinary Optimization of LCCC Package**
- 亮点：Pareto front + 敏感性分析图组合
- 值得学：多学科设计优化的权衡可视化

### 10.7 算法/求解器流程图 — 推荐论文

**论文：PACT — Extensible Parallel Thermal Simulator**
- 亮点：软件架构模块图，plugin 机制展示，extensibility 清晰
- 值得学：仿真器软件架构的专业展示

**论文：ARO — Autoregressive Operator Learning with Active Learning**
- 亮点：多阶段 pipeline 流程，active learning 循环图
- 值得学：ML training pipeline + decision loop 的组合

**论文：Parameterized Thermal Simulation Based on PINN**
- 亮点：PINN 训练流程，physics residual 可视化
- 值得学：Physics-informed ML 的标准流程图

### 10.8 芯片布局/placement 图 — 推荐论文

**论文：ATPlace2.5D — Analytical Thermal-aware Chiplet Placement**
- 亮点：Chiplet 布局图，热热点可视化，初始 vs. 优化对比
- 值得学：Placement 优化的前后对比可视化

**论文：TAP-2.5D — Thermally-aware Chiplet Placement**
- 亮点：2.5D 系统俯视图，organic substrate / interposer 标注
- 值得学：Chiplet 系统的层次可视化

### 10.9 GPU/并行加速图 — 推荐论文

**论文：GPU Acceleration of High-Precision Stochastic Solver**
- 亮点：GPU 架构图（CUDA block/thread），并行化策略可视化
- 值得学：并行计算可视化的专业画法

### 10.10 PCM（相变材料）热分析图 — 推荐论文

**论文：Transient Thermal Characterization of Power Module with PCM**
- 亮点：DSC 曲线，相变焓可视化，红外热像叠加
- 值得学：PCM 表征数据（热流 vs. 温度）的规范展示

---

## 十一、持续更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-04-01 | 初始版本：基于40篇论文阅读，标注逻辑图为主，涵盖各类图表 |
| 2026-04-01 | 新增第十章：热管理论文优秀图例清单，基于57篇 thermal journals 分析 |
| | 新增：ML架构图、热图、热阻网络、封装剖面图、transient曲线、优化图、算法流程图、chiplet placement、GPU并行、PCM热分析 |
| | 来源：DeepOHeat, ARO, PINN, GAN, FNO, Point Cloud Transformer, 3D-ICE, FaStTherm, MatEx, NSGA-II, PCM, PACT 等 |
