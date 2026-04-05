# INTRODUCTION

## A. Background and Motivation

Three-dimensional integrated circuits (3D-IC) stack multiple active dies with interposer layers and through-silicon vias (TSVs), enabling continued performance scaling in microelectronics [1], [2]. This vertical integration, however, concentrates power density and restricts heat dissipation paths—thermal management has become one of the dominant design constraints [3], [4]. Localized hot spots from non-uniform heat dissipation threaten both device reliability and circuit performance [5].

Designers need accurate temperature predictions throughout the design cycle, from early floorplanning to final packaging optimization. The problem is that detailed finite element (FEA) or finite difference (FDM) simulations take hours or days per configuration, which makes it impractical when thousands of design alternatives must be evaluated.

Compact thermal models came along as a faster alternative. HotSpot (2006) uses resistance-capacitance (RC) networks to approximate heat flow, cutting computation time by orders of magnitude while remaining suitable for early-stage decisions [6]. 3D-ICE (2010) builds on this idea with a Fickian heat transfer model and compact FD scheme, delivering 975× speedup over full FEA [7]. MatEx (2015) adds a matrix exponential formulation for sensitivity analysis [8]. These models work well when heat spreading is approximately uniform and thermal resistances stay roughly constant. But when power densities become non-uniform and boundary conditions grow complex—common in modern 3D-IC stacks—these assumptions break down and prediction error increases [9].

## B. Machine Learning for Thermal Simulation

Machine learning entered thermal simulation around 2020 to handle arbitrary component configurations without manually tuned thermal resistances. ThermGAN (2020) uses a conditional GAN to generate temperature maps from power layouts, reaching 0.47°C RMSE in-distribution with interactive floorplanning speed [10]. One issue is that ThermGAN needs a separate trained model for each floorplan topology; generalizing to new component arrangements means retraining from scratch.

Operator learning caught on because it learns a direct mapping from input parameters to temperature fields, analogous to solving an infinite-dimensional function [11]. DeepOHeat (2023) applies the DeepONet trunk-branch architecture, showing 1,000× to 300,000× speedup over FEA with R² > 0.99 in-distribution [12]. DeepOHeat has three specific limitations. First, its MLP-based trunk networks exhibit spectral bias—they capture low-frequency thermal patterns well but struggle with sharp gradients near heat sources [13]. Second, physics-informed training requires computing second-order derivatives across millions of collocation points, making training memory-intensive for high-resolution problems. Third, like most operator learning frameworks, DeepOHeat gives no mechanism to tell whether a prediction is trustworthy at unseen configurations [13].

Several works have tried to address these weaknesses. DeepOHeat-v1 (2026) replaces MLP basis functions with Kolmogorov-Arnold networks (KAN), cutting error by 1.25× to 6.29× and speeding up training by 70.6× through separable forward derivatives [13]. 3D CoSim (2025) adds EM-thermal coupling to DeepONet with a multi-input architecture, reaching 844× to 7,600× speedup with 99.8% thermal accuracy [14]. FSA-Heat (2025) processes temperature fields in frequency and spatial domains simultaneously [15]. Therm-PCT (2025) uses point cloud transformers and reports zero-shot generalization to 5.7× larger component counts [16]. HeatDiffUNet (2025) applies diffusion models, getting down to 2.31°C error with only 200 samples [17]. SAU-FNO (2025) combines self-attention with U-Net and FNO, reaching 842× speedup [18].

The issue is that all these methods are evaluated almost exclusively on in-distribution configurations—models trained on N-component systems are tested on the same or similar N-component configurations [12], [13], [14], [15], [16], [17], [18]. This masks whether the models can actually generalize. For thermal prediction, changing the number or arrangement of heat sources fundamentally alters the temperature field topology. A model that scores R² > 0.99 during testing can fail catastrophically when queried on a configuration it has never seen.

## C. Physics-Informed Operator Learning

Physics-informed neural networks (PINNs) encode governing equations directly into the training loss, using automatic differentiation to penalize violations of the heat equation and boundary conditions at any point in the domain [19], [20]. This reduces reliance on large datasets and provides physical consistency. ThermPINN (2023) demonstrates this approach for full-chip thermal analysis, achieving 6× speedup over conventional solvers with 0.47 K MAE [21]. The catch is that ThermPINN requires retraining when geometric parameters change; extending it to arbitrary floorplans means a new training run for each layout.

The idea here is that combining physics-informed loss with operator learning should improve out-of-distribution generalization. Heat conduction physics does not depend on how many heat sources exist—the steady-state Laplace equation and Robin boundary conditions hold regardless of configuration. These constraints should act as a domain-independent prior, keeping the model from producing physically implausible predictions at unseen component counts.

To test this, a physics-informed operator learning framework augments the SetFNOModel—a Transformer encoder plus Fourier Neural Operator decoder—with two physics loss terms. The PDE loss enforces ∇²T = 0 at interior points away from heat sources, corresponding to Laplace's equation for steady-state conduction. The boundary condition loss enforces a Robin condition T_edge = c_adj × T_adj (c_adj = 0.536) at domain edges, approximating convective cooling to ambient. These losses are weighted by λ_pde = 0.001 and λ_bc, where λ_bc controls the strength of boundary physics constraints.

Results show that physics constraints reshape how the model behaves on unseen configurations. Without them, out-of-distribution performance degrades badly as component count increases. With them, predictions stay physically plausible because the model gets penalized for violating heat conduction physics, even at configurations never seen during training.

## D. Contributions

This paper makes three contributions:

1) A physics-informed operator learning framework trained on 1-5 component data (300 samples total) that achieves R² > 0.90 for 6-8 component test cases, showing practical generalization without requiring massive training sets.

2) Evidence that physics-informed loss prevents catastrophic OOD failure. When tested on 9-component configurations at 30 W total power, the physics-informed model maintains R² = 0.55 while a purely data-driven baseline collapses to R² = 0.08.

3) A systematic λ_bc sensitivity analysis showing that the optimal weight is 0.0005. Weaker constraints (0.0001) provide insufficient physics bias; stronger ones (0.001, 0.01) overconstrain the model and reduce both in-distribution and OOD performance.

## E. Paper Organization

Section II covers thermal modeling background and related work. Section III describes the proposed framework, including architecture and physics-informed loss design. Section IV explains the experimental setup. Section V presents results on generalization, ablation, and parameter sensitivity. Section VI discusses implications and limitations. Section VII concludes.
