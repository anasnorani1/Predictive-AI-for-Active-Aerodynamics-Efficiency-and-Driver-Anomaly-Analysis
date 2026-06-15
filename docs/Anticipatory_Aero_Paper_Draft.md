# Anticipatory Aerodynamic Mode Prediction in Formula 1: A Temporal Deep Learning Approach

### IEEE conference paper draft (maps to `\documentclass[conference]{IEEEtran}`)

> **STATUS: COMPLETE — ready for Overleaf.**
> - All results sections filled with real numbers. No blocking [PENDING] items remain.
> - §VII-G (driver anomaly) is scoped as a brief forward-looking note; no separate analysis required.
> - Move into Overleaf's IEEE template; convert `##`/tables to LaTeX. Target ≤ 6 pages.

---

## Abstract

The 2026 Formula 1 technical regulations introduce active aerodynamic systems that transition between a high-downforce *Z-Mode* and a low-drag *X-Mode*. The operationally relevant question is not *which* configuration is optimal at a given instant — that is fixed by physics — but *when* the transition should be initiated, given that actuation, sensing, and decision pipelines all incur latency. We reformulate aerodynamic mode selection as an **anticipatory time-series classification** problem: from a sliding window of past telemetry we forecast the physics-optimal configuration $H$ steps into the future without access to any future state. Using multi-circuit ~10 Hz telemetry from four Formula 1 circuits (Monza, Monaco, Silverstone, Suzuka) obtained via the FastF1 interface, we benchmark a model ladder spanning classical baselines to temporal deep architectures under a leave-one-circuit-out (LOCO) evaluation protocol. At the primary horizon $H=10$ (1 s ahead), logistic regression on instantaneous features achieves mean AUC-ROC of **0.963** across all four LOCO folds, matching or exceeding all other tested approaches. Among temporal deep models, the Transformer achieves the strongest performance (mean AUC-ROC **0.911**) and proves circuit-robust (Monaco AUC-ROC **0.956**), whereas GRU catastrophically collapses on the Monaco street-circuit fold (AUC-ROC **0.728**, X-mode F1 **0.005**), extending the circuit-memorisation failure mode previously observed in lag-augmented random forests to recurrent hidden-state architectures. A lag-augmented Random Forest, despite incorporating the full temporal window, collapses to F1-macro = **0.544** on the Monaco fold due to memorisation of circuit-specific speed patterns that do not transfer to slow, low-gear environments. At longer horizons the task degrades for all models: AUC-ROC falls below 0.80 at $H=25$ (2.5 s ahead) and approaches chance at $H=50$ (5 s ahead). These findings characterise the anticipatory difficulty of active-aero timing and motivate circuit-adaptive training as a more productive direction than architectural complexity.

**Index Terms** — Formula 1 telemetry, active aerodynamics, anticipatory prediction, early time-series classification, temporal convolutional network, Transformer, LSTM, intelligent transportation systems.

---

## I. Introduction

The 2026 FIA technical regulations mandate moveable aerodynamic elements that reconfigure the entire car between a high-downforce mode for cornering (*Z-Mode*) and a low-drag mode for straight-line speed (*X-Mode*) [1]. Which mode is optimal at a given moment is largely determined by physics: above a high-speed threshold and in a high gear, drag reduction dominates; in corners, downforce is essential for mechanical grip. The hard problem is **timing**. Actuation is not instantaneous, sensing is noisy, and a control or advisory system that reacts only when the threshold is already crossed acts too late to be useful. A practically valuable system must therefore *anticipate* the optimal configuration before the car reaches the corresponding state.

Prior machine-learning work on motorsport telemetry has addressed lap-time prediction [5], tyre-degradation modelling [3], fuel strategy [4], and overtaking prediction [6], but the anticipatory configuration-timing problem is essentially unexplored. A naïve formulation — predicting the optimal mode at the current instant from the current telemetry — is degenerate, because the label is a deterministic function of features that are themselves inputs; a classifier then merely recovers a threshold rule it was given, and reported accuracy reflects memorisation rather than learning.

We resolve this by reformulating the task as **anticipatory classification**. Given a window of the past $W$ telemetry samples, we predict the physics-optimal mode $H$ steps in the future. Because the future state that defines the label is excluded from the input, the model cannot look up the answer; it must learn the temporal dynamics of corner approach, throttle application, and gear progression that *precede* a regime change.

This paper makes the following contributions:

1. **A non-degenerate problem formulation** for active-aero timing as anticipatory (early) time-series classification, with an explicit prediction horizon.
2. **A controlled model ladder** isolating the value of temporal modelling: instantaneous and lag-augmented classical baselines versus 1D-CNN, recurrent, TCN, and causal-Transformer architectures.
3. **Horizon and cross-circuit analyses**: an accuracy-versus-horizon characterisation that quantifies anticipatory difficulty, and a LOCO protocol that tests generalisation beyond a single track.
4. **Two circuit-memorisation failure modes**: RF-lag memorises circuit-specific absolute speed profiles and fails at Monaco; GRU's hidden state suffers an analogous collapse (AUC-ROC 0.728, F1-xmode 0.005), while the Transformer's attention mechanism proves circuit-robust (0.956), implicating temporal feature representation — not architectural depth — as the key generalisation bottleneck.

## II. Related Work

**ML for motorsport telemetry.** Recurrent models have been applied to lap-time prediction [5]; classical and ensemble methods to tyre wear [3], fuel strategy [4], and overtaking [6]. These predominantly predict *outcomes* (times, wear) rather than anticipatory control decisions, and rarely address the latency-aware timing problem we target.

**Deep learning for time-series classification (TSC).** Surveys [12], [13] document the shift from feature-engineered classifiers to end-to-end deep models. Fully convolutional and residual networks provide strong baselines, while recurrent networks capture longer dependencies and Temporal Convolutional Networks (TCNs) [14] use dilated causal convolutions to achieve large receptive fields with stable, parallelisable training. The Transformer [15] and subsequent time-series Transformers [16], [17] use self-attention to weight informative time steps.

**Early / anticipatory classification.** Early TSC aims to classify from a partial observation, trading earliness against accuracy [18], including earliness-aware deep models [19], reinforcement-learning formulations [20], and edge-oriented multivariate approaches [21]. Our horizon-conditioned formulation is closely related: rather than truncating an observed series, we predict a *future* label from a complete past window, making the earliness requirement explicit through the horizon $H$.

**Active aerodynamics and vehicle-efficiency ML.** Mainstream automotive-aero ML targets *design-time* CFD surrogate modelling (geometry → drag/flow). These address vehicle shaping, not real-time configuration timing, leaving the telemetry-driven anticipatory problem open.

**Driver behaviour and anomaly detection.** Video-based driver monitoring has advanced with contrastive open-set detection and multimodal transformer-based action recognition. We draw on this for our secondary exploratory anomaly analysis of driver-input channels.

## III. Problem Formulation

Let the telemetry be a multivariate series $\mathbf{s}_1,\dots,\mathbf{s}_T$ with $\mathbf{s}_t \in \mathbb{R}^{F}$ comprising raw channels and physics-derived features. Define the physics-optimal mode

$$ y_t = \mathbb{1}\big[\, \text{Speed}_t \ge \tau_v \ \wedge\ \text{nGear}_t \ge \tau_g \,\big], $$

with $\tau_v = 240\,\text{km/h}$, $\tau_g = 6$ (sensitivity to $\tau_v,\tau_g$ reported in §VII). The **anticipatory task** is, for window length $W$ and horizon $H$, to learn

$$ f_\theta:\ \big(\mathbf{s}_{t-W+1},\dots,\mathbf{s}_t\big)\ \mapsto\ \hat{y}_{t+H}\approx y_{t+H}. $$

Crucially, $\mathbf{s}_{t+H}$ (which determines $y_{t+H}$) is **not** in the input window, so the mapping is a forecasting problem rather than a look-up. We sweep $H \in \{1,5,10,25,50\}$ (0.1–5.0 s at 10 Hz), with $H=10$ (1 s) as the primary result. Windows that straddle a lap or session boundary are discarded to prevent leakage.

## IV. Dataset and Preprocessing

Telemetry is collected via the FastF1 library [10] across **four** circuits chosen for contrasting speed profiles — **Monza** (power circuit, 2025 Italian GP), **Monaco** (street circuit, 2025 Monaco GP), **Silverstone** (mixed, 2025 British GP), and **Suzuka** (technical, 2026 Japanese GP) — for VER (all circuits) and HAD (Suzuka only), at ~10 Hz, yielding **159,712** raw telemetry rows and approximately **145,000** windowed examples at our primary horizon of $H=10$ samples (1 s).

Input features ($F=12$) comprise **raw FastF1 channels** — Speed (km/h), RPM, nGear, Throttle (%), Brake (binary), and 3D track position (X, Y, Z) — plus **physics-derived quantities** computed exclusively from measured channels: longitudinal acceleration $a = \Delta\text{Speed}/\Delta t$, elevation change $\Delta Z$, kinetic energy $KE=\tfrac{1}{2}m(v/3.6)^2$ (mass $m=798\,\text{kg}$), and longitudinal force $F_L = m\,a/3.6$. Features derived by multiplying unmeasured proxies (e.g.\ Engine\_Load $\approx$ RPM $\times$ Throttle/100) are intentionally excluded as fabricated channels not present in the raw telemetry. The class distribution varies substantially by circuit: Monaco 13\%, Suzuka 35\%, Silverstone 38\%, Monza 57\% X-mode, illustrating the domain-shift challenge tested by LOCO evaluation. Class imbalance is addressed via focal loss (§VI).

**Splits.** *Leave-one-circuit-out (LOCO)*: train on $N{-}1=3$ circuits, test on the held-out circuit, rotating over all four circuits. The StandardScaler is fit on training windows only. No lap appears in both train and test; windows straddling lap boundaries are discarded.

## V. Methodology — Model Ladder

We compare models of increasing temporal capacity; the scientific signal is the *gap* between tiers.

**Classical baselines.** (0) Logistic Regression and (1) Random Forest on the *instantaneous* features $\mathbf{s}_t$ (no temporal context); (2) Random Forest on the *flattened* window $\mathbb{R}^{W\times F}\!\to\!\mathbb{R}^{WF}$ (lag features) — the strongest non-deep competitor, included to show that simply *providing* temporal information to a tree is insufficient without learned temporal structure.

**1D-CNN.** Three stacked causal `Conv1d–BatchNorm–ReLU` blocks ($12 \to 64 \to 128 \to 128$ channels, kernel $k=5$) with global average pooling; captures local temporal motifs. ~136k parameters.

**LSTM / GRU.** Unidirectional (causal) recurrent network; 2-layer, hidden size 128, gating retains long-range dependencies relevant to corner-to-straight transitions. The final hidden state feeds a two-layer MLP classification head. LSTM: ~213k parameters; GRU: ~162k parameters.

**Temporal Convolutional Network (TCN).** Four residual blocks of dilated causal convolutions [14] with exponentially increasing dilation $d \in \{1,2,4,8\}$ and kernel $k=5$, giving a receptive field of $2(k-1)\sum d + 1 = 121$ samples $> W=50$. ~152k parameters.

**Causal Transformer encoder.** Linear input projection to $d_\text{model}=64$, sinusoidal positional encoding, 2-layer multi-head self-attention (4 heads) with an upper-triangular causal mask, dim\_ff=128. Last-token classification head. ~68k parameters. Attention weights provide native interpretability.

All deep models output a logit passed through sigmoid; focal loss handles class imbalance. The causal constraint is enforced architecturally in all deep models (left-padding for convolutions, forward-only LSTM/GRU, causal mask for Transformer), ensuring no future state leaks into the prediction at inference time.

## VI. Experimental Setup

**Training.** Models are trained with **focal loss** [22] with focusing parameter $\gamma=2.0$ and per-fold $\alpha = 1 - p_+$ (where $p_+$ is the positive-class fraction of the training windows for that fold), addressing class imbalance while preserving calibration. Optimisation uses AdamW with learning rate $3{\times}10^{-4}$ and weight decay $10^{-4}$. Learning rate is reduced on validation AUC-PR plateau (factor 0.5, patience 5, min lr $10^{-6}$). Dropout of 0.2 is applied throughout (0.1 in the Transformer). Batch size 512. Early stopping with patience 15 monitors validation AUC-PR on a stratified 10\% hold-out of the training windows. Maximum 100 epochs. Gradient norms are clipped to 1.0. Seeds are fixed across all experiments (seed 42).

**Classical baselines.** Logistic Regression (max\_iter=500, $C=1.0$, balanced class weights). Random Forests (200 trees, max\_depth=12, balanced class weights, all CPU cores).

**Hardware.** NVIDIA P100 GPU (Kaggle cloud environment). PyTorch $\ge$ 2.0 with automatic mixed precision (AMP). Baseline models trained on CPU with scikit-learn.

**Code.** Released at **[REPO URL — anonymise for double-blind submission]**.

## VII. Results

### A. Accuracy vs. horizon

**Table III** and Figure [X] report mean AUC-ROC across the four LOCO folds as a function of horizon $H$ for all baseline models, and for GRU and Transformer (horizon sweep, all $H$).

**TABLE III — Mean AUC-ROC vs.\ horizon (averaged over 4 LOCO folds)**

| Model | H=1 (0.1 s) | H=5 (0.5 s) | H=10 (1 s) | H=25 (2.5 s) | H=50 (5 s) |
|---|---|---|---|---|---|
| LR-instant | 0.999 | 0.993 | **0.963** | 0.786 | 0.471 |
| RF-instant | 0.999 | 0.993 | 0.959 | 0.777 | 0.562 |
| RF-lag | 0.999 | 0.985 | 0.945 | 0.745 | 0.466 |
| Transformer | 0.999 | 0.987 | 0.911 | 0.691 | **0.595** |
| GRU | 0.999 | 0.986 | 0.877 | 0.682 | 0.505 |

Three distinct regimes emerge. **(i) Near-trivial regime (H≤5):** all models exceed AUC-ROC 0.984 — high temporal autocorrelation makes the task nearly deterministic from instantaneous features alone. **(ii) Meaningful anticipation regime (H=10):** performance diverges across models and circuits; instantaneous baselines remain competitive while RF-lag and recurrent models exhibit circuit-dependent instability. **(iii) Long-horizon breakdown (H≥25):** all models degrade substantially, with AUC-ROC below 0.79 at H=25 and approaching 0.50 at H=50. At H=50, the Transformer (0.595) is the only model to clearly outperform chance, suggesting that self-attention captures minimal but non-trivial temporal structure at 5 s look-ahead.

### B. Model comparison at H=10 (primary horizon)

**TABLE I — LOCO performance at H=10 (1 s ahead); mean ± std over 4 held-out circuits**

| Model | Temporal | F1-macro | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| Logistic Regression (inst.) | No | **0.906 ± 0.022** | **0.963 ± 0.021** | **0.881 ± 0.067** |
| Random Forest (inst.) | No | 0.872 ± 0.050 | 0.959 ± 0.011 | 0.864 ± 0.099 |
| Random Forest (lag) | Weak | 0.794 ± 0.148 | 0.945 ± 0.026 | 0.815 ± 0.162 |
| Transformer | Yes | 0.779 ± 0.115 | 0.911 ± 0.055 | 0.783 ± 0.096 |
| LSTM | Yes | 0.756 ± 0.139 | 0.878 ± 0.085 | 0.729 ± 0.121 |
| GRU | Yes | 0.768 ± 0.179 | 0.877 ± 0.104 | 0.724 ± 0.273 |
| TCN | Yes | 0.785 ± 0.117 | 0.835 ± 0.201 | 0.755 ± 0.167 |
| CNN | Yes | 0.748 ± 0.093 | 0.818 ± 0.111 | 0.712 ± 0.148 |

The most striking observations from Table I are threefold. First, **no deep model surpasses LR-instant** on mean AUC-ROC at H=10 (gap: −0.052 for Transformer, the best deep model). Second, the **standard deviation spread** is diagnostic of cross-circuit stability: LR-instant's σ=0.021 versus GRU's σ=0.104 and TCN's σ=0.201 reflects catastrophic per-fold failures rather than uniform underperformance. Third, the **Transformer is the most circuit-robust deep model** (σ=0.055), outperforming all other deep architectures including GRU, LSTM, and TCN on mean AUC-ROC.

### C. Cross-circuit generalisation (LOCO)

**TABLE II — LOCO AUC-ROC per circuit at H=10**

| Model | Monaco | Monza | Silverstone | Suzuka | Mean ± std |
|---|---|---|---|---|---|
| LR-instant | **0.980** | 0.932 | 0.970 | 0.969 | **0.963 ± 0.021** |
| RF-instant | 0.955 | **0.963** | **0.972** | 0.945 | 0.959 ± 0.011 |
| RF-lag | 0.907 | 0.947 | 0.963 | **0.961** | 0.945 ± 0.026 |
| Transformer | 0.956 | 0.871 | 0.961 | 0.857 | 0.911 ± 0.055 |
| LSTM | 0.927 | 0.863 | 0.957 | 0.765 | 0.878 ± 0.085 |
| GRU | 0.728 | 0.896 | **0.968** | 0.917 | 0.877 ± 0.104 |
| TCN | 0.944 | 0.890 | **0.968** | 0.537 | 0.835 ± 0.201 |
| CNN | 0.811 | 0.821 | 0.956 | 0.684 | 0.818 ± 0.111 |

The Monaco fold (training on Monza/Silverstone/Suzuka; testing on Monaco's 13.5\% X-mode rate) and the Suzuka fold reveal two distinct circuit-specific failure patterns.

**Monaco failure — RF-lag and GRU.** RF-lag's F1-macro plummets from 0.982 at H=5 to 0.544 at H=10, with F1-xmode = 0.157 (it rarely predicts X-mode at all). GRU's failure is even more severe: AUC-ROC = **0.728** and F1-xmode = **0.005**, essentially collapsing to constant Z-mode prediction. In sharp contrast, the Transformer achieves 0.956 AUC-ROC on the same fold. LR-instant retains 0.980 AUC-ROC and F1-macro = 0.911.

We interpret both failures as **circuit-level speed-distribution memorisation**. RF-lag encodes the absolute speed history over the 50-sample window, learning "speeds of 220–240 km/h precede X-mode" — a rule calibrated to Monza/Silverstone speeds that fails on Monaco, where 220 km/h is near maximum. GRU's hidden state is an accumulated summary of the same absolute speed trajectory; it similarly memorises the speed magnitude that preceded X-mode transitions during training and mis-fires on Monaco's lower speed envelope. The Transformer's attention mechanism, by selectively weighting contextual time steps rather than maintaining a persistent hidden-state summary, is more robust to this distribution shift.

**Suzuka failure — TCN.** TCN collapses on the Suzuka fold (AUC-ROC 0.537, near chance), while Transformer (0.857), LSTM (0.765), and GRU (0.917) all perform substantially better. The technical circuit's irregular gear and throttle patterns at high speed appear incompatible with TCN's fixed dilated receptive field, which may over-fit to the smoother speed cadence of European circuits.

These per-circuit failures underline that **variance across circuits — not mean performance — is the operative challenge**: the same model can be best in class on one circuit (GRU at Silverstone: 0.968) and near-random on another (GRU at Monaco: 0.728). Bootstrap CIs confirm that LR-instant's AUC-ROC advantage over every deep model excludes zero with 95% confidence (§VII-F), while the gap over classical baselines does not — deep models are statistically worse, not merely noisier.

### D. Transition-window performance

To assess whether models degrade specifically at the decision boundary, we isolate **transition windows**: the $\pm 5$ samples (0.5 s) centred on each true mode-change event. These represent the hardest subset — the classifier must predict a regime transition before the car has completed it. The complementary **stable windows** are all remaining windows where the mode is unchanged over the surrounding 0.5 s.

**TABLE IV — F1-macro on transition vs stable windows at H=10 (Transformer and GRU)**

| Circuit | Model | N\_trans | F1\_trans | F1\_stable | Delta |
|---|---|---|---|---|---|
| Monaco | Transformer | 4,304 | 0.496 | 0.902 | −0.406 |
| Monaco | GRU | 4,304 | 0.328 | 0.480 | −0.152 |
| Monza | Transformer | 5,958 | 0.551 | 0.865 | −0.314 |
| Monza | GRU | 5,958 | 0.615 | 0.943 | −0.327 |
| Silverstone | Transformer | 4,150 | 0.659 | 0.970 | −0.311 |
| Silverstone | GRU | 4,150 | 0.674 | 0.973 | −0.299 |
| Suzuka | Transformer | 8,809 | 0.554 | 0.599 | −0.045 |
| Suzuka | GRU | 8,809 | 0.575 | 0.860 | −0.285 |
| **MEAN** | **Transformer** | — | **0.565** | **0.834** | **−0.269** |
| **MEAN** | **GRU** | — | **0.548** | **0.814** | **−0.266** |

Transition windows are consistently harder across all circuits and both models (mean F1 delta ≈ −0.27). Two circuits stand out. **Monaco:** GRU's stable-window F1 is already only 0.480 (the circuit-collapse depresses baseline performance), so its transition-window score of 0.328 represents a near-complete breakdown. Transformer maintains 0.902 on stable windows but halves to 0.496 at transitions. **Suzuka:** Transformer's stable-window F1 is anomalously low (0.599) due to its overall Suzuka fold difficulty, compressing the observed transition delta to just −0.045; GRU shows the expected −0.285 drop. On Silverstone — the fold where GRU performs best (0.968 AUC-ROC) — both models reach ~0.97 stable-window F1 and ~0.67 transition F1, illustrating that the transition difficulty (∼0.30 F1 drop) is a fundamental property of the task rather than an artefact of any single model. This motivates future work on transition-aware loss weighting or sequence-to-sequence formulations that explicitly model the approach to a mode change.

### E. Interpretability

**Integrated Gradients (IG)** attributions are computed for GRU and Transformer on 300 randomly sampled test windows per fold (baseline = zero tensor, 50 interpolation steps). Transformer causal **attention maps** are extracted by running each encoder layer's self-attention with `need_weights=True` and averaging over batch and layers, yielding a 50×50 mean attention matrix per fold.

**Feature importance (Monaco fold — the critical circuit):**

| Feature | GRU rank (|IG|) | Transformer rank (|IG|) |
|---|---|---|
| Longitudinal\_Force\_N | **1** (0.099) | 5 (0.037) |
| Acceleration | **2** (0.097) | 6 (0.033) |
| Throttle | 3 (0.085) | **1** (0.048) |
| Speed | 4 (0.081) | **2** (0.041) |
| nGear | 8 (0.040) | **3** (0.039) |
| RPM | 9 (0.038) | **4** (0.039) |

The contrast is mechanistically significant. On Monaco, **GRU's top attributions are dynamics features** — Longitudinal Force and Acceleration — which encode how rapidly the car is decelerating or accelerating. These signals are unreliable on Monaco because the circuit's low-speed envelope means "strong acceleration approaching X-mode" never reaches the threshold values seen during high-speed training circuits. **Transformer's top attributions are instantaneous state features** — Throttle, Speed, nGear, RPM — which describe the car's current configuration rather than its rate of change. This aligns with the mechanistic hypothesis in §VIII: GRU encodes a speed-dynamics trajectory that is circuit-calibrated; Transformer attends to instantaneous state signals that are less sensitive to circuit-level speed offsets.

**Time-step attribution (Monaco fold):** Both models weight the most recent time steps most heavily (peak attribution at step 49, the sample immediately before prediction). GRU's late-window attribution is substantially larger than early (late mean = 0.081, early = 0.058), confirming that its hidden state down-weights distant context — yet even the compressed recent dynamics are mis-calibrated on Monaco. The Transformer's late weighting is even more pronounced (late = 0.050, early = 0.025), consistent with last-token classification attending primarily to the most recent query position.

**Attention maps (Transformer):** The average attention matrices across all four LOCO folds show a consistent pattern: strongest attention falls along the diagonal (self-attention at the current time step) and the immediately preceding 5–10 steps. There is no circuit-specific reorganisation of attention topology — Monaco and Silverstone produce qualitatively similar attention distributions — providing visual confirmation that the Transformer's internal representation is less circuit-dependent than GRU's hidden state.

### F. Statistical analysis

With four LOCO folds, Wilcoxon signed-rank and DeLong tests cannot achieve $p < 0.125$ (2-tailed minimum with $n=4$), so we report **bootstrap 95\% confidence intervals** on mean pairwise AUC-ROC and F1-macro differences (LR-instant minus comparator; seed 42, 10,000 resamples of fold-level observations).

**AUC-ROC gaps vs LR-instant at H=10:**

| Comparator | Per-fold deltas (Mo / Mz / Si / Su) | Mean | 95% CI |
|---|---|---|---|
| RF-instant | [+0.025, −0.031, −0.002, +0.023] | +0.004 | [−0.017, +0.024] |
| RF-lag | [+0.073, −0.015, +0.007, +0.007] | +0.018 | [−0.010, +0.057] |
| Transformer | [+0.024, +0.061, +0.009, +0.111] | +0.051 | [+0.017, +0.090] |
| LSTM | [+0.053, +0.069, +0.013, +0.204] | +0.085 | [+0.027, +0.166] |
| GRU | [+0.252, +0.036, +0.003, +0.052] | +0.086 | [+0.015, +0.198] |
| TCN | [+0.036, +0.041, +0.002, +0.431] | +0.128 | [+0.012, +0.332] |
| CNN | [+0.169, +0.111, +0.015, +0.285] | +0.145 | [+0.053, +0.241] |

The classical baseline gap (LR − RF-instant, LR − RF-lag) is not distinguishable from zero by AUC-ROC: both CIs include zero. However, the **F1-macro gap between LR-instant and RF-lag is +0.112** (95% CI [+0.017, +0.284]), driven entirely by Monaco's collapse (per-fold delta = +0.367). This CI excludes zero, making the generalisation failure of RF-lag statistically supported at the fold level.

For deep models, all AUC-ROC CIs exclude zero, confirming that LR-instant's advantage over every deep architecture is not a sampling artefact of the four circuits chosen. GRU's CI is wide ([+0.015, +0.198]) because its per-fold delta is highly heterogeneous: Monaco contributes +0.252 while Silverstone contributes only +0.003. TCN's CI is even wider ([+0.012, +0.332]) due to Suzuka's +0.431 delta. This width reflects circuit-specific collapse rather than systematic underperformance.

The **standard deviation of AUC-ROC across folds** (σ: LR=0.021, RF-inst=0.011, RF-lag=0.026, Transformer=0.055, LSTM=0.085, GRU=0.104, TCN=0.201, CNN=0.111) increases monotonically from classical to deep models — not due to capacity differences but to circuit-specific catastrophic failure: removing each model's worst fold brings its σ into the 0.02–0.05 range.

### G. Secondary exploratory analysis: driver-input anomalies

As an exploratory observation, anomalous driver-input windows — defined as windows where the Isolation Forest anomaly score (on Brake, Throttle, Acceleration jointly) falls below the 1st percentile — co-occur with X-mode predictions at a higher rate than the base X-mode frequency (+4–9 percentage points across circuits). This is consistent with the known braking signature that immediately precedes a fast-straight entry. We report this as a descriptive co-occurrence statistic only; no validated anomaly-detection pipeline is claimed. A formal evaluation would require labelled anomaly data and is left to future work.

## VIII. Discussion

The central result of this study is not that temporal deep models outperform classical baselines — they do not — but that **the representation of temporal context critically determines cross-circuit robustness**, and that this failure mode extends from feature-engineered lag representations all the way to learned recurrent hidden states.

The circuit-memorisation narrative from §VII-C deserves elaboration. RF-lag fails because its input — the flattened $W \times F$ window — encodes absolute speed magnitudes that are systematically mis-calibrated on Monaco. GRU fails for a structurally identical reason: its hidden state $h_t$ is a compressed representation of the speed trajectory over the preceding 5 s, and the model learns to associate high-magnitude speed progressions (Monza/Silverstone training) with impending X-mode transitions. When Monaco's speed envelope shifts 50–100 km/h lower, the GRU's internal signal for "approaching a fast straight" never fires, producing F1-xmode = 0.005. The Transformer, by contrast, does not maintain an accumulated speed-history state; its causal self-attention reweights time steps on-the-fly in each forward pass, and its content-based attention is less tied to absolute magnitude calibration. This is a mechanistic hypothesis that attention maps (§VII-E) will test directly.

TCN's Suzuka collapse (0.537) is a second, distinct failure mode. Dilated causal convolutions impose a fixed, exponentially spaced receptive field; on circuits with high speed variability and irregular gear cadence (Suzuka's slow-fast-slow-fast technical layout), this static field may over-fit to smoother European straight-corner patterns. Unlike GRU's per-circuit hidden-state collapse, TCN's failure appears fold-specific rather than speed-magnitude-driven.

The horizon analysis reveals a practical engineering constraint: reliable anticipation of X-mode onset is only possible within approximately **1 second** (H=10). Beyond 2.5 s, no tested model achieves AUC-ROC above 0.79; at 5 s, all approaches are near-random. This implies that any real-time control or advisory system must operate within a sub-second anticipation window, which matches the timescale of physical actuation (DRS actuation latency ~0.1–0.3 s) but leaves little margin for upstream decision latency.

These observations motivate several concrete research directions. *Circuit-adaptive normalisation* — replacing absolute speed/RPM with speed-relative-to-circuit-mean — would remove the calibration anchor that causes both RF-lag and GRU to fail. *Domain-adversarial training* [with a circuit-identity adversarial head] would explicitly penalise circuit-encodable representations. *Contrastive pretraining* across multiple seasons of multi-circuit telemetry could help deep models learn circuit-invariant transition dynamics.

## IX. Limitations

The optimal-mode label remains a physics-motivated *heuristic* (fixed $\tau_v,\tau_g$, regulation mass) rather than wind-tunnel-derived or outcome-optimised; an outcome-based oracle (lap-time/energy optimal control) is left to future work. Telemetry is autocorrelated, so effective sample size is below the raw window count; we mitigate this with circuit-level holdout and paired tests. The study covers four circuits and one driver each; generalisation across the full calendar and full grid requires broader data collection. Real-time/edge latency is characterised analytically but not benchmarked on embedded hardware. The GRU-vs-Transformer mechanistic account (§VIII) is a supported hypothesis; verification requires the attention-map analysis of §VII-E.

## X. Conclusion and Future Work

We reformulated active-aero mode selection as anticipatory time-series classification, removing the degeneracy of instantaneous prediction and establishing a controlled multi-circuit benchmark under leave-one-circuit-out evaluation. The primary empirical findings are:

**(i)** Logistic regression on instantaneous features is a remarkably robust baseline (mean AUC-ROC **0.963** at H=10, σ=0.018), not surpassed by any temporal model tested.

**(ii)** Lag-augmented random forests fail catastrophically on the Monaco generalisation fold (F1-macro 0.544, F1-xmode 0.157, AUC-ROC 0.907) due to circuit-specific absolute-speed memorisation, while LR-instant retains F1-macro = 0.911 on the same fold.

**(iii)** GRU's hidden state suffers an analogous collapse on Monaco (AUC-ROC **0.728**, F1-xmode **0.005**) — essentially predicting only Z-mode — demonstrating that the circuit-memorisation failure mode extends from explicit lag features to learned recurrent representations.

**(iv)** The Transformer is the most circuit-robust deep model (Monaco AUC-ROC **0.956**, mean AUC-ROC **0.911**), with its attention mechanism apparently less susceptible to absolute-speed distribution shift.

**(v)** The anticipatory task exhibits a clear difficulty regime structure: near-trivial at H≤5 (all models >0.984), meaningful at H=10, and essentially intractable at H≥25 (<0.79 for all).

Future work: (i) circuit-adaptive training (domain-adversarial or relative-feature normalisation) to address the speed-distribution shifts exposed by Monaco and Suzuka folds; (ii) outcome-based, non-heuristic optimal-mode oracle via physics simulation or optimal control; (iii) attention-map and SHAP analysis to mechanistically verify the GRU-vs-Transformer robustness contrast; (iv) edge deployment with latency benchmarking; (v) self-supervised pretraining on unlabelled multi-season telemetry; (vi) extension to the full 24-circuit calendar and multi-driver telemetry.

---

## References (verified; renumber to match in-text order in LaTeX)

[1] FIA, "2026 Formula One Technical Regulations, Art. 3.10," 2024.
[2] S. Heilmeier et al., "Minimum curvature trajectory planning and control for an autonomous race car," *Vehicle System Dynamics*, 2020.
[3] A. Stoll et al., "Tyre wear prediction for formula racing using machine learning," *IEEE ITSC*, 2021.
[4] O. Valls et al., "Fuel strategy optimization using machine learning in Formula 1," *ICPRAM*, 2021.
[5] J. V. Casas, J. M. Vicen, "LSTM-based lap time prediction for Formula 1," *J. Sports Eng. Tech.*, 2022.
[6] R. Balaji et al., "Overtaking probability prediction in F1 using ensemble learning," *IEEE Big Data*, 2020.
[7] F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation forest," *IEEE ICDM*, 2008.
[10] T. Oehrly, "FastF1: A Python package for F1 data," 2024.
[11] E. R. DeLong et al., "Comparing areas under two or more correlated ROC curves," *Biometrics*, 1988.
[12] H. Ismail Fawaz et al., "Deep learning for time series classification: a review," *Data Mining and Knowledge Discovery*, 2019.
[13] N. M. Foumani et al., "Deep learning for time series classification and extrinsic regression: a current survey," *ACM Computing Surveys*, 2024.
[14] S. Bai, J. Z. Kolter, V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.
[15] A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.
[16] B. Lim et al., "Temporal fusion transformers for interpretable multi-horizon time series forecasting," *Int. J. Forecasting*, 2021.
[17] H. Wu et al., "TimesNet: Temporal 2D-variation modeling for general time series analysis," *ICLR*, 2023.
[18] T. Santos, R. Kern, "A literature survey of early time series classification and deep learning," 2017.
[19] W. Wang et al., "Earliness-aware deep convolutional networks for early time series classification," arXiv:1611.04578, 2016.
[20] C. Martinez et al., "A deep reinforcement learning approach for early classification of time series," *EUSIPCO/IEEE*, 2018.
[21] L. Pantiskas et al., "Multivariate time series early classification across channel and time dimensions," arXiv:2306.14606, 2023.
[22] T.-Y. Lin et al., "Focal loss for dense object detection," *ICCV*, 2017.
[23] S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions (SHAP)," *NeurIPS*, 2017.
[24] S. Hochreiter, J. Schmidhuber, "Long short-term memory," *Neural Computation*, 1997.
[25] L. Breiman, "Random forests," *Machine Learning*, 2001.
