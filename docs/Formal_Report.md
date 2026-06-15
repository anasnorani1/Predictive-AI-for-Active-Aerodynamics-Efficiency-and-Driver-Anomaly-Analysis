# Predictive AI for Active Aerodynamics Efficiency and Driver Anomaly Analysis
## A Formal Technical Report

---

**Submitted by:** Anas Norani · Hanan Majeed · Maha Mohsin  
**Program:** Bachelor of Science in Data Science  
**Department:** School of Electrical Engineering and Computer Science (SEECS)  
**University:** National University of Sciences and Technology (NUST), Islamabad  
**Submission Date:** June 2026  
**Supervisor:** [Supervisor Name]

---

## Abstract

This report presents a comprehensive two-phase machine learning study applied to Formula 1 car telemetry data. **Phase 1** investigates driver anomaly detection and current aerodynamic (aero) state classification using a single-race dataset from the 2026 Japanese Grand Prix at Suzuka (63,673 rows). Using Isolation Forest, K-Means clustering, Logistic Regression, and Random Forest, we establish that the current aero mode is perfectly classifiable from instantaneous sensor readings (Random Forest AUC-ROC = 1.000), while anomalous driving windows — predominantly maximum-braking and gear-shift events — exhibit statistically significant feature deviations (Cohen's d up to −1.41). **Phase 2** extends this to the harder problem of *anticipatory prediction*: given a 5-second history of car telemetry, predict the aero mode **1 second into the future** on circuits the model has never trained on. We benchmark eight models under Leave-One-Circuit-Out (LOCO) evaluation across four circuits (Monaco, Monza, Silverstone, Suzuka; 159,712 rows). Logistic Regression on instantaneous features achieves the best mean AUC-ROC of **0.963**, while the Transformer achieves the strongest deep-learning performance (**0.911**). A critical finding is the catastrophic collapse of the Gated Recurrent Unit (GRU) at the Monaco circuit (AUC-ROC = 0.728, F1-Xmode = 0.005), caused by circuit-level speed-distribution memorisation — a failure mode also observable in the Temporal Convolutional Network at Suzuka (AUC-ROC = 0.537). All models fail beyond a 2.5-second horizon, constraining practical systems to sub-second anticipation windows. The unifying insight across both phases is that models which encode circuit-specific speed baselines fail to generalise — a property that directly connects Phase 1's anomaly detection to Phase 2's cross-circuit breakdown.

**Keywords:** Formula 1 telemetry, active aerodynamics, anomaly detection, anticipatory prediction, leave-one-circuit-out, time-series classification, Transformer, GRU, circuit memorisation.

---

## Table of Contents

1. Introduction and Motivation
2. Dataset Description
3. Phase 1 — Driver Anomaly Detection and Aero State Classification
   - 3.1 Feature Engineering
   - 3.2 Anomaly Detection (Isolation Forest)
   - 3.3 Driving Zone Clustering (K-Means)
   - 3.4 Aero State Classification
   - 3.5 Statistical Analysis
4. Phase 2 — Anticipatory Aero Mode Prediction
   - 4.1 Problem Formulation
   - 4.2 Models and Experimental Setup
   - 4.3 Primary Results at H=10 (1 Second Ahead)
   - 4.4 Per-Circuit LOCO Breakdown
   - 4.5 Horizon Sweep Analysis
   - 4.6 Transition Window Analysis
   - 4.7 Feature Interpretability (Integrated Gradients)
   - 4.8 Transformer Attention Maps
   - 4.9 Statistical Significance
5. Discussion
6. Conclusions
7. Future Work
8. References

---

## 1. Introduction and Motivation

The 2026 Formula 1 technical regulations (FIA Art. 3.10) introduce mandatory Active Aerodynamic systems that physically reconfigure a car's wing geometry between two operating modes:

- **Z-Mode (High Downforce):** Wings fully deployed to maximise mechanical grip through corners.
- **X-Mode (Low Drag):** Wings retracted to reduce aerodynamic resistance on fast straights.

The transition rule is governed by a fixed threshold: the car enters X-Mode when its speed exceeds **240 km/h** while in **gear 6 or higher**. While the optimal mode at any given moment is determined by physics, the challenge is one of **timing**: actuation, sensing, and onboard decision pipelines all incur latency on the order of 0.1–0.3 seconds. A control system that reacts only when the threshold is already crossed operates too late to be fully effective.

This project investigates whether machine learning can solve this timing problem in two steps:

**Phase 1** — Before predicting the future, we must understand the present. Using data from the 2026 Suzuka Grand Prix, we ask: (i) Can the current aero mode be identified from sensor data? (ii) Are there detectable anomalies in driver behaviour? (iii) What do different driving regimes look like statistically?

**Phase 2** — Building on Phase 1's feature understanding, we ask the harder question: can a model predict the aero mode **1 second before** the car reaches that state, on circuits it has never seen during training?

---

## 2. Dataset Description

All telemetry data was collected via the **FastF1** open-source Python library from official FIA timing feeds at approximately 10 Hz (10 samples per second per car).

### 2.1 Phase 1 Dataset

| Property | Value |
|---|---|
| Race | 2026 Japanese Grand Prix |
| Circuit | Suzuka International Racing Course |
| Drivers | Max Verstappen (RB), Isack Hadjar (RB) |
| Total rows | 63,673 |
| Sampling rate | ~10 Hz |
| X-Mode proportion | 34.7% |
| Z-Mode proportion | 65.3% |

### 2.2 Phase 2 Multi-Circuit Dataset

| Circuit | Race | Rows | X-Mode % | Speed Profile |
|---|---|---|---|---|
| Monaco | 2025 Monaco GP | 39,881 | 13.3% | Slow street (max ~220 km/h) |
| Monza | 2025 Italian GP | 31,098 | 56.9% | High-speed power (max ~365 km/h) |
| Silverstone | 2025 British GP | 25,060 | 38.9% | Mixed (max ~310 km/h) |
| Suzuka | 2026 Japanese GP | 63,673 | 34.7% | Technical (max ~330 km/h) |
| **Total** | | **159,712** | **~36%** | |

The four circuits were chosen to maximise diversity in speed profiles, corner types, and X-Mode prevalence — intentionally creating a challenging generalisation benchmark. Monaco's X-Mode rate of only 13.3% (versus Monza's 56.9%) is the single largest source of domain shift in the study.

### 2.3 Aero Mode Label Definition

```
y[t] = 1  (X-Mode)  if  Speed[t] >= 240 km/h  AND  nGear[t] >= 6
y[t] = 0  (Z-Mode)  otherwise
```

In Phase 2, the label uses the **future** time step: `y[t+H]`, where H is the prediction horizon. The model never sees future data in its input window.

---

## 3. Phase 1 — Driver Anomaly Detection and Aero State Classification

### 3.1 Feature Engineering

Starting from raw FastF1 channels (Speed, RPM, nGear, Throttle, Brake, X/Y/Z position), the following additional features were computed:

| Feature | Formula / Source | Rationale |
|---|---|---|
| Acceleration (m/s²) | ΔSpeed / Δtime | Longitudinal dynamics |
| Elevation_Delta (m) | ΔZ / Δtime | Track gradient signal |
| Kinetic_Energy_MJ | ½ × 798 kg × (Speed/3.6)² / 10⁶ | Physics-grounded energy |
| Longitudinal_Force_N | 798 × Acceleration / 3.6 | Force proxy |
| High_Speed_Zone | 1 if Speed≥240 AND nGear≥6 | Aero mode indicator |
| Heavy_Braking | 1 if Brake≥50% AND Accel≤−5 m/s² | Braking event flag |
| Gear_Shift_Active | 1 if consecutive gear change within 0.5 s | Gear transition flag |

**Descriptive statistics (Phase 1 dataset, n=63,673):**

| Feature | Mean | Std | Min | 25th % | Median | 75th % | Max | Skewness |
|---|---|---|---|---|---|---|---|---|
| Speed (km/h) | 218.2 | 65.4 | 58.0 | 173.5 | 216.0 | 272.0 | 347.0 | −0.21 |
| RPM | 10,619 | 811 | 6,252 | 10,175 | 10,741 | 11,175 | 12,708 | −0.96 |
| nGear | 5.48 | 1.90 | 1 | 4 | 5 | 7 | 8 | −0.40 |
| Throttle (%) | 62.5 | 42.5 | 0 | 13.0 | 90.6 | 100.0 | 104.0 | −0.47 |
| Acceleration (m/s²) | 0.01 | 4.85 | −28.0 | −0.81 | 0.00 | 1.40 | 28.0 | −0.81 |
| Elevation_Delta (m) | 0.00 | 4.34 | −82.0 | −1.00 | 0.01 | 1.13 | 108.5 | −0.07 |
| Kinetic_Energy (MJ) | 1.60 | 0.86 | 0.10 | 0.93 | 1.44 | 2.28 | 3.71 | +0.31 |

### 3.2 Anomaly Detection — Isolation Forest

**Method:** Isolation Forest with contamination=5%, 100 estimators, random_state=42. Features used: Speed, RPM, nGear, Throttle, Brake, Acceleration, Elevation_Delta.

**Result:** Approximately **5% of rows (≈3,184 samples)** flagged as anomalous.

**Feature comparison — anomalous vs. normal windows:**

| Feature | Anomalous Mean | Normal Mean | Cohen's d | Effect Size | MW p-value |
|---|---|---|---|---|---|
| Acceleration (m/s²) | −6.20 | +0.34 | **−1.41** | Large | <0.001 |
| Throttle (%) | 22.7 | 64.6 | **−1.01** | Large | <0.001 |
| Brake | 0.686 | 0.157 | **+1.44** | Large | <0.001 |
| Gear_Shift_Active | 0.711 | 0.022 | **+3.96** | Large | <0.001 |
| Heavy_Braking | 0.290 | 0.000 | **+2.86** | Large | <0.001 |

All differences are highly statistically significant (Mann-Whitney U, p < 0.001 for all). The anomaly signature is: **near-zero throttle, hard braking, strong negative acceleration, and active gear shifting** — characteristic of late-braking traction limit events at technical corners.

**Association with aero mode:** Chi-squared test (χ² = 817.4, p = 9.0×10⁻¹⁸⁰, OR = 0.18 [95% CI: 0.16–0.20]) confirms anomalous windows are strongly associated with Z-Mode. Anomalous windows are only **18% as likely** to be in X-Mode as normal windows — anomalies occur almost exclusively during braking events, not on straights.

**Geographical analysis:** Anomalies mapped to GPS coordinates cluster at Suzuka's known technical sections: the Esses complex (Turns 3–7), Degner curves (Turns 8–9), the Hairpin (Turn 11), Spoon Curve (Turn 13), and the 130R–Casio chicane approach.

### 3.3 Driving Zone Clustering — K-Means

**Method:** K-Means with k=4, determined by elbow method on within-cluster sum of squares. Features normalised before clustering: Speed, nGear, Acceleration, Brake, Throttle.

**Statistical validation:** Kruskal-Wallis H-test on Speed across four clusters: **H = 34,439, p < 0.001, η² = 0.555**.  
Cluster membership explains **55.5% of variance in Speed** — confirming clusters are physically meaningful. All six pairwise Dunn comparisons are significant at p < 0.001 (Bonferroni corrected).

**Cluster characterisation:**

| Cluster | Dominant Behaviour | Mean Speed (km/h) | Mean Throttle (%) | Aero Mode |
|---|---|---|---|---|
| 0 | High-speed aero phase | 295 ± 28 | 98 ± 4 | X-Mode (dominant) |
| 1 | Medium cornering | 210 ± 35 | 71 ± 22 | Z-Mode |
| 2 | Heavy braking and entry | 165 ± 42 | 12 ± 19 | Z-Mode |
| 3 | Slow hairpin approach | 112 ± 38 | 8 ± 15 | Z-Mode |

### 3.4 Aero State Classification

**Method:** Both Logistic Regression (L2, C=1.0) and Random Forest (100 trees, balanced class weights) trained on instantaneous features. 80/20 stratified train/test split. Evaluated using bootstrap confidence intervals (10,000 resamples, seed=42).

**Results:**

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC 95% CI |
|---|---|---|---|---|---|---|
| **Random Forest** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | [1.000, 1.000] |
| Logistic Regression | 0.979 | 0.881 | 1.000 | 0.937 | 0.9997 | [0.9993, 0.9999] |

**Interpretation:** Both models achieve near-perfect performance because the aero mode label is defined by thresholds on Speed and nGear — features that are present in the input. This result validates the feature set and confirms the task is physically sound, but it does not demonstrate genuine predictive capability. This directly motivates Phase 2, where the label is shifted H=10 steps into the future and the model cannot simply look up the answer.

**DeLong AUC comparison:** ΔAUC = 0.0003, z = −2.06, p = 0.039. RF is statistically but not practically superior to LR at the boundary. McNemar test: RF wins on 274 edge-case samples vs. LR's 0 (χ² = 272, p ≈ 0), indicating RF handles the non-linear speed/gear boundary more precisely.

### 3.5 Statistical Analysis

#### 3.5.1 Normality Tests

All 11 features were tested with Shapiro-Wilk (subsample n=5,000) and D'Agostino-Pearson K². **All features are significantly non-normal (p < 0.001 for both tests on all features).** This rules out parametric methods and motivates bootstrap CIs and non-parametric tests throughout.

| Feature | SW Statistic | SW p | D-P Statistic | D-P p | Verdict |
|---|---|---|---|---|---|
| Speed | 0.978 | <0.001 | 4,541.6 | <0.001 | Non-normal |
| RPM | 0.961 | <0.001 | 9,037.1 | <0.001 | Non-normal |
| nGear | 0.928 | <0.001 | 3,247.7 | <0.001 | Non-normal |
| Throttle | 0.768 | <0.001 | 358,737 | <0.001 | Non-normal |
| Acceleration | 0.796 | <0.001 | 16,482.7 | <0.001 | Non-normal |
| Elevation_Delta | 0.576 | <0.001 | 24,598.6 | <0.001 | Non-normal |
| Kinetic_Energy | 0.962 | <0.001 | 11,485.9 | <0.001 | Non-normal |

#### 3.5.2 Driver Comparison — Verstappen vs. Hadjar

| Test | Statistic | p-value | Effect Size | Interpretation |
|---|---|---|---|---|
| Mann-Whitney U on Speed | U = 0.0 | <0.001 | Cohen's d = 0.042 | Significant but small |
| Chi-squared on aero deviation rate | χ² = 73.1 | <0.001 | Cramér's V = 0.113 | Small association |

Both differences are statistically significant but practically small (d < 0.1, V < 0.2). Verstappen shows marginally higher average Speed and a slightly different X-Mode entry rate. Driver differences exist but are not large enough to warrant driver-specific models for Phase 2.

#### 3.5.3 Permutation Feature Importance

| Rank | Feature | Importance | Std | 95% CI | Significant |
|---|---|---|---|---|---|
| 1 | **nGear** | **0.167** | 0.006 | [0.155, 0.180] | YES |
| 2 | **Elevation_Delta** | **0.162** | 0.004 | [0.156, 0.170] | YES |
| 3 | **Brake** | **0.081** | 0.003 | [0.075, 0.086] | YES |
| 4 | Speed | 0.0004 | 0.0001 | [0.000, 0.0005] | YES |
| 5–18 | All other features | ≈ 0.000 | ≈ 0 | — | NO |

**Key observation:** nGear outranks Speed as the most important feature. This reflects that nGear is the harder constraint: maintaining high gear requires both high speed *and* appropriate track position, whereas speed alone can be high in many configurations. Elevation_Delta's high importance is Suzuka-specific — the circuit's varied terrain creates reliable gear-zone boundaries. This finding directly informs the feature interpretation in Phase 2 (Section 4.7).

---

## 4. Phase 2 — Anticipatory Aero Mode Prediction

### 4.1 Problem Formulation

The anticipatory prediction task is formally defined as:

**Input:** A sliding window of W=50 time steps ending at time t:  
`X = [x_{t-49}, x_{t-48}, ..., x_t]` with shape (50, 12)

**Target:** Aero mode label H=10 steps into the future:  
`y[t+10] = 1 if Speed[t+10] >= 240 AND nGear[t+10] >= 6`

The **label index is strictly greater than the last input index**. The future state that defines the label is never available to the model. This formulation resolves the degeneracy identified in Phase 1 — the model cannot simply recover the threshold rule from its inputs.

**Window parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| W | 50 samples | 5-second input window |
| Primary H | 10 samples | 1-second look-ahead |
| Horizon sweep | H ∈ {1, 5, 10, 25, 50} | 0.1s to 5.0s ahead |
| F | 12 features | See Section 4.2 |

**Feature set (F=12):**

| Category | Features |
|---|---|
| Raw telemetry | Speed, RPM, nGear, Throttle, Brake, X, Y, Z |
| Physics-derived | Acceleration, Elevation_Delta, Kinetic_Energy_MJ, Longitudinal_Force_N |

Note: Engine_Load (= RPM×Throttle/100) is a fabricated proxy absent from raw telemetry and is intentionally excluded.

**Evaluation protocol — Leave-One-Circuit-Out (LOCO):**

| Fold | Train Circuits | Test Circuit |
|---|---|---|
| 1 | Monza + Silverstone + Suzuka | Monaco |
| 2 | Monaco + Silverstone + Suzuka | Monza |
| 3 | Monaco + Monza + Suzuka | Silverstone |
| 4 | Monaco + Monza + Silverstone | Suzuka |

StandardScaler is fit on **training windows only** — no test-circuit statistics leak into normalisation.

### 4.2 Models and Experimental Setup

**Model ladder (ordered by temporal capacity):**

| Model | Architecture | Parameters | Temporal Reasoning |
|---|---|---|---|
| LR-instant | Logistic Regression on x_t only | ~12 | None |
| RF-instant | Random Forest on x_t only | ~50k | None |
| RF-lag | Random Forest on flattened 50×12 window | ~50k | Weak (explicit lags) |
| CNN | Causal Conv1D × 3 blocks, global avg pooling | ~136k | Local patterns |
| CausalLSTM | 2-layer unidirectional LSTM, hidden=128 | ~213k | Long-range recurrent |
| CausalGRU | 2-layer unidirectional GRU, hidden=128 | ~162k | Long-range recurrent |
| TCN | 4 dilated causal residual blocks (d=1,2,4,8) | ~152k | Fixed receptive field |
| CausalTransformer | 2-layer encoder, 4 heads, causal mask | ~68k | Self-attention |

**Training configuration (all deep models):**

| Hyperparameter | Value |
|---|---|
| Loss function | Focal Loss (γ=2.0, α=1−pos_rate per fold) |
| Optimiser | AdamW (lr=3×10⁻⁴, weight_decay=10⁻⁴) |
| LR schedule | Cosine annealing + ReduceOnPlateau (patience=5) |
| Batch size | 512 |
| Dropout | 0.2 (0.1 for Transformer) |
| Early stopping | Patience=15, monitors AUC-PR on 10% val split |
| Max epochs | 100 |
| Gradient clipping | norm ≤ 1.0 |
| Reproducibility | seed=42 |
| Hardware | NVIDIA P100 GPU, 16 GB VRAM (Kaggle cloud) |

### 4.3 Primary Results at H=10 (1 Second Ahead)

**Table 1 — Mean ± std across 4 LOCO folds at H=10 (primary horizon)**

| Model | Temporal | F1-Macro | AUC-ROC | AUC-PR | σ AUC-ROC |
|---|---|---|---|---|---|
| **LR-instant** | No | **0.906 ± 0.022** | **0.963 ± 0.021** | **0.881 ± 0.067** | **0.021** |
| RF-instant | No | 0.872 ± 0.050 | 0.959 ± 0.011 | 0.864 ± 0.099 | 0.011 |
| RF-lag | Weak | 0.794 ± 0.148 | 0.945 ± 0.026 | 0.815 ± 0.162 | 0.026 |
| **Transformer** | Yes | 0.779 ± 0.115 | **0.911 ± 0.055** | 0.783 ± 0.096 | 0.055 |
| LSTM | Yes | 0.756 ± 0.139 | 0.878 ± 0.085 | 0.729 ± 0.121 | 0.085 |
| GRU | Yes | 0.768 ± 0.179 | 0.877 ± 0.104 | 0.724 ± 0.273 | 0.104 |
| TCN | Yes | 0.785 ± 0.117 | 0.835 ± 0.201 | 0.755 ± 0.167 | 0.201 |
| CNN | Yes | 0.748 ± 0.093 | 0.818 ± 0.111 | 0.712 ± 0.148 | 0.111 |

**Key observations from Table 1:**

1. **No deep model surpasses LR-instant.** The gap between LR-instant and the best deep model (Transformer) is 0.052 AUC-ROC points. This is not a marginal difference — it is statistically confirmed (Section 4.9).

2. **Standard deviation is diagnostic.** LR-instant's σ = 0.021 versus GRU's σ = 0.104 and TCN's σ = 0.201 reflects catastrophic per-fold failure, not uniform underperformance. A model with high σ is not consistently bad — it is excellent on some circuits and collapses on others.

3. **Transformer is the most circuit-robust deep model** (σ = 0.055), outperforming all other deep architectures on both mean AUC-ROC and cross-circuit stability.

### 4.4 Per-Circuit LOCO Breakdown

**Table 2 — AUC-ROC per held-out circuit at H=10**

| Model | Monaco | Monza | Silverstone | Suzuka | Mean ± std |
|---|---|---|---|---|---|
| **LR-instant** | **0.980** | 0.932 | 0.970 | 0.969 | **0.963 ± 0.021** |
| RF-instant | 0.955 | **0.963** | **0.972** | 0.945 | 0.959 ± 0.011 |
| RF-lag | 0.907 | 0.947 | 0.963 | **0.961** | 0.945 ± 0.026 |
| **Transformer** | 0.956 | 0.871 | 0.961 | 0.857 | 0.911 ± 0.055 |
| LSTM | 0.927 | 0.863 | 0.957 | 0.765 | 0.878 ± 0.085 |
| **GRU** | **0.728** | 0.896 | 0.968 | 0.917 | 0.877 ± 0.104 |
| TCN | 0.944 | 0.890 | 0.968 | **0.537** | 0.835 ± 0.201 |
| CNN | 0.811 | 0.821 | 0.956 | 0.684 | 0.818 ± 0.111 |

**Table 3 — F1-Macro per held-out circuit at H=10**

| Model | Monaco | Monza | Silverstone | Suzuka | Mean ± std |
|---|---|---|---|---|---|
| **LR-instant** | **0.911** | 0.873 | 0.935 | 0.905 | **0.906 ± 0.026** |
| RF-instant | 0.803 | 0.902 | 0.934 | 0.851 | 0.872 ± 0.059 |
| RF-lag | 0.544 | 0.838 | 0.923 | 0.870 | 0.794 ± 0.168 |
| Transformer | 0.806 | 0.802 | 0.914 | 0.596 | 0.779 ± 0.133 |
| LSTM | 0.770 | 0.832 | 0.895 | 0.528 | 0.756 ± 0.157 |
| **GRU** | **0.465** | 0.874 | 0.919 | 0.812 | 0.768 ± 0.196 |
| TCN | 0.760 | 0.849 | 0.924 | 0.609 | 0.785 ± 0.139 |
| CNN | 0.784 | 0.772 | 0.841 | 0.594 | 0.748 ± 0.106 |

**Table 4 — F1-Xmode (X-Mode class only) per held-out circuit at H=10**

| Model | Monaco | Monza | Silverstone | Suzuka |
|---|---|---|---|---|
| **LR-instant** | 0.849 | 0.903 | 0.922 | 0.879 |
| RF-instant | 0.650 | 0.920 | 0.917 | 0.797 |
| RF-lag | **0.157** | 0.863 | 0.904 | 0.824 |
| Transformer | 0.660 | 0.848 | 0.896 | 0.376 |
| LSTM | 0.594 | 0.861 | 0.877 | 0.256 |
| **GRU** | **0.005** | 0.901 | 0.903 | 0.740 |
| TCN | 0.572 | 0.881 | 0.908 | 0.409 |
| CNN | 0.619 | 0.838 | 0.826 | 0.376 |

**Critical failure analysis:**

The **GRU at Monaco** is the most severe failure in the study. With AUC-ROC = 0.728 and F1-Xmode = **0.005**, the model predicts X-Mode on fewer than 0.5% of samples where it is required. This is not a weak classifier — it is essentially a constant Z-Mode predictor. On the same circuit with the same data, Transformer achieves AUC-ROC = 0.956 and LR-instant achieves 0.980. The failure is model-specific, not data-specific.

The **TCN at Suzuka** (AUC-ROC = 0.537, near random chance) is a second, independent failure mode. Unlike GRU's circuit-speed collapse, TCN's failure at Suzuka appears linked to the circuit's irregular gear/throttle cadence at high speed, which is incompatible with TCN's fixed dilated receptive field trained predominantly on smoother European straight-corner patterns.

### 4.5 Horizon Sweep Analysis

**Table 5 — Mean AUC-ROC vs. prediction horizon (4 LOCO folds)**

| Model | H=1 (0.1s) | H=5 (0.5s) | H=10 (1s) | H=25 (2.5s) | H=50 (5s) |
|---|---|---|---|---|---|
| LR-instant | 0.999 | 0.993 | 0.963 | 0.786 | 0.471 |
| RF-instant | 0.999 | 0.993 | 0.959 | 0.777 | 0.562 |
| RF-lag | 0.999 | 0.985 | 0.945 | 0.745 | 0.466 |
| Transformer | 0.999 | 0.987 | 0.911 | 0.691 | **0.595** |
| GRU | 0.999 | 0.986 | 0.877 | 0.682 | 0.505 |

**Table 6 — GRU per-circuit AUC-ROC across all horizons**

| Circuit | H=1 | H=5 | H=10 | H=25 | H=50 |
|---|---|---|---|---|---|
| Monaco | 0.9996 | 0.9894 | 0.728 | 0.693 | 0.804 |
| Monza | 0.9993 | 0.9777 | 0.896 | 0.635 | 0.296 |
| Silverstone | 0.9994 | 0.9907 | 0.968 | 0.818 | 0.565 |
| Suzuka | 0.9992 | 0.9873 | 0.917 | 0.582 | 0.356 |

**Table 7 — Transformer per-circuit AUC-ROC across all horizons**

| Circuit | H=1 | H=5 | H=10 | H=25 | H=50 |
|---|---|---|---|---|---|
| Monaco | 0.9995 | 0.9951 | 0.956 | 0.791 | 0.825 |
| Monza | 0.9995 | 0.9860 | 0.871 | 0.696 | 0.397 |
| Silverstone | 0.9993 | 0.9912 | 0.961 | 0.849 | 0.687 |
| Suzuka | 0.9991 | 0.9758 | 0.857 | 0.628 | 0.470 |

**Horizon regime analysis:**

Three regimes are clearly visible across Tables 5–7:

**Trivial regime (H ≤ 5, ≤ 0.5s):** All models exceed AUC-ROC 0.975. High temporal autocorrelation makes the task nearly deterministic — knowing the car's current state is sufficient to predict with high confidence what state it will be in 0.5 seconds later. These results are not informative of model capability.

**Meaningful anticipation regime (H = 10, 1s):** Performance diverges substantially. The gap between best (LR-instant, 0.963) and worst (CNN, 0.818) exceeds 14 AUC-ROC points. Circuit-specific failures emerge (GRU at Monaco drops from 0.989 at H=5 to 0.728 at H=10). This is the primary evaluation horizon.

**Long-horizon breakdown (H ≥ 25, ≥ 2.5s):** All models degrade to below AUC-ROC 0.79. At H=50 (5 seconds ahead), LR-instant achieves only 0.471 (below chance) and GRU Monza achieves only 0.296. Only the Transformer retains marginally above-chance performance (0.595 at H=50), suggesting its self-attention mechanism captures some minimal long-range dependency.

**Engineering implication:** A practical active-aero control system must operate within approximately 1-second anticipation. No current machine-learning method is reliable enough for predictive control beyond 2.5 seconds.

### 4.6 Transition Window Analysis

Mode changes are the most safety-critical and operationally relevant prediction events. We isolate **transition windows**: the ±5 samples (±0.5 seconds) centred on each true mode-change event.

**Table 8 — F1-Macro: transition vs. stable windows at H=10**

| Circuit | Model | N Transition | F1 Transition | F1 Stable | Δ F1 |
|---|---|---|---|---|---|
| Monaco | Transformer | 4,304 | 0.496 | 0.902 | **−0.406** |
| Monaco | GRU | 4,304 | 0.328 | 0.480 | −0.152 |
| Monza | Transformer | 5,958 | 0.551 | 0.865 | −0.314 |
| Monza | GRU | 5,958 | 0.615 | 0.943 | −0.327 |
| Silverstone | Transformer | 4,150 | 0.659 | 0.970 | −0.311 |
| Silverstone | GRU | 4,150 | 0.674 | 0.973 | −0.299 |
| Suzuka | Transformer | 8,809 | 0.554 | 0.599 | −0.045 |
| Suzuka | GRU | 8,809 | 0.575 | 0.860 | −0.285 |
| **MEAN** | **Transformer** | — | **0.565** | **0.834** | **−0.269** |
| **MEAN** | **GRU** | — | **0.548** | **0.814** | **−0.266** |

**Interpretation:** A mean F1 drop of ≈ −0.27 at transition windows is observed for both models across all circuits. This is a **fundamental property of the anticipatory task**: at the ±0.5 second boundary of a mode change, the car's telemetry has not yet fully committed to the new regime, making it impossible for any model to predict with high confidence. This result holds even on high-performing circuits (GRU at Silverstone: stable F1 = 0.973, transition F1 = 0.674).

**Monaco anomaly:** GRU's stable-window F1 at Monaco is already only 0.480 (the circuit collapse described in Section 4.4 depresses the baseline). Its transition F1 of 0.328 represents a total breakdown. Transformer, by contrast, achieves 0.902 on stable Monaco windows despite the same distribution shift.

### 4.7 Feature Interpretability — Integrated Gradients

**Method:** Integrated Gradients (IG) with baseline = zero tensor, 50 interpolation steps, computed on 300 randomly sampled test windows per fold.

**Table 9 — Feature importance: GRU vs. Transformer (Monaco fold, H=10)**

| Feature | GRU |IG| rank | GRU |IG| value | Transformer |IG| rank | Transformer |IG| value |
|---|---|---|---|---|
| Longitudinal_Force_N | **#1** | 0.099 | #5 | 0.037 |
| Acceleration | **#2** | 0.097 | #6 | 0.033 |
| Throttle | #3 | 0.085 | **#1** | 0.120 |
| Speed | #4 | 0.081 | **#2** | 0.102 |
| nGear | #8 | 0.040 | **#3** | 0.096 |
| RPM | #9 | 0.038 | **#4** | 0.078 |
| Brake | #5 | 0.074 | #7 | 0.026 |
| Kinetic_Energy | #6 | 0.066 | #8 | 0.022 |
| Elevation_Delta | #7 | 0.052 | #9 | 0.019 |
| X (position) | #10 | 0.033 | #10 | 0.017 |
| Y (position) | #11 | 0.028 | #11 | 0.015 |
| Z (position) | #12 | 0.021 | #12 | 0.013 |

**Mechanistic interpretation:**

GRU's top-ranked features at Monaco are **dynamics features** — Longitudinal Force and Acceleration describe *how fast* the car is changing speed. During training on Monza and Silverstone, these signals fire strongly in the seconds before a fast straight (high positive acceleration approaching X-Mode). At Monaco, this pattern never reaches the trained threshold — the circuit's lower speed envelope means "strong positive acceleration to 250 km/h" never occurs. GRU's prediction engine is calibrated to a speed envelope it never encounters at Monaco.

Transformer's top-ranked features are **instantaneous state features** — Throttle, Speed, nGear, RPM describe *what the car is doing right now*. These signals transfer across circuits because the X-Mode threshold rule (Speed ≥ 240 AND nGear ≥ 6) is identical at Monaco, Monza, Silverstone, and Suzuka. A car approaching that threshold looks the same instantaneously regardless of the circuit's overall speed level.

This contrast mechanistically explains the performance gap in Table 2.

**Time-step attribution (Monaco fold):**

| Model | Late-window mean |IG| (steps 40–49) | Early-window mean |IG| (steps 0–9) | Ratio |
|---|---|---|---|
| GRU | 0.081 | 0.058 | 1.40× |
| Transformer | 0.050 | 0.025 | 2.00× |

Both models weight recent time steps more heavily than early ones. Transformer is more strongly focused on recent context (2× ratio), consistent with last-token classification attending primarily to its immediate neighbourhood. GRU's smaller ratio suggests its hidden state retains some early-window context — the compressed speed trajectory from training circuits that misfires at Monaco.

### 4.8 Transformer Attention Maps

Average self-attention weights (50×50 matrix) are extracted per fold by running each encoder layer with `need_weights=True` and averaging over batch, layers, and test samples.

**Key finding:** The attention topology is **consistent across all four LOCO folds**. Monaco and Silverstone produce qualitatively similar average attention matrices: strongest weights along the diagonal (current time step attending to itself) and the immediately preceding 5–10 steps, with a secondary cluster at the very first time step (long-range context). There is **no circuit-specific reorganisation** of attention patterns — the model does not learn to attend differently at Monaco versus Monza. This is the attention-map confirmation that Transformer's internal representation is less circuit-dependent than GRU's hidden state.

### 4.9 Statistical Significance

With only n=4 LOCO folds, standard paired tests (Wilcoxon signed-rank, DeLong) cannot achieve p < 0.125 (two-tailed minimum with n=4). We therefore report **bootstrap 95% confidence intervals** on mean pairwise AUC-ROC differences (LR-instant minus comparator; 10,000 resamples of fold-level observations, seed=42).

**Table 10 — Bootstrap CI for AUC-ROC gap vs. LR-instant at H=10**

| Comparator | Per-fold deltas (Mo / Mz / Si / Su) | Mean Gap | 95% CI | Interpretation |
|---|---|---|---|---|
| RF-instant | [+0.025, −0.031, −0.002, +0.023] | +0.004 | [−0.017, +0.024] | CI includes zero — not significant |
| RF-lag | [+0.073, −0.015, +0.007, +0.007] | +0.018 | [−0.010, +0.057] | CI includes zero — not significant |
| Transformer | [+0.024, +0.061, +0.009, +0.111] | +0.051 | [**+0.017, +0.090**] | CI excludes zero — **significant** |
| LSTM | [+0.053, +0.069, +0.013, +0.204] | +0.085 | [**+0.027, +0.166**] | CI excludes zero — **significant** |
| GRU | [+0.252, +0.036, +0.003, +0.052] | +0.086 | [**+0.015, +0.198**] | CI excludes zero — **significant** |
| TCN | [+0.036, +0.041, +0.002, +0.431] | +0.128 | [**+0.012, +0.332**] | CI excludes zero — **significant** |
| CNN | [+0.169, +0.111, +0.015, +0.285] | +0.145 | [**+0.053, +0.241**] | CI excludes zero — **significant** |

**Summary of statistical findings:**

- LR-instant is **not statistically distinguishable** from RF-instant and RF-lag by AUC-ROC alone (CIs include zero). The three classical models are statistically equivalent in aggregate performance.
- LR-instant is **statistically significantly better** than all five deep learning models (all AUC-ROC CIs exclude zero). The advantage is not a sampling artefact of the four circuits chosen.
- The wide CIs for GRU ([+0.015, +0.198]) and TCN ([+0.012, +0.332]) reflect **circuit-specific catastrophic failure** rather than uniform underperformance — a model that scores +0.252 at Monaco and +0.003 at Silverstone is not consistently bad.

---

## 5. Discussion

### 5.1 Why the Simplest Model Wins

The result that Logistic Regression on 12 instantaneous features (≈12 parameters) outperforms a GRU with 162,000 parameters is counterintuitive. The explanation is **circuit memorisation**.

Under LOCO evaluation, a model that encodes circuit-specific patterns from training data will fail when those patterns do not transfer to the test circuit. Consider what each model memorises:

**RF-lag** receives the full 5-second speed history. It learns: *"When speed has been rising through 220–250 km/h over the past 5 seconds, X-Mode is imminent."* This rule is correct at Monza (where speed regularly reaches 340 km/h) and Silverstone (280 km/h). At Monaco, where maximum speed barely exceeds 220 km/h, this rule never fires — and RF-lag collapses to F1-Xmode = 0.157.

**GRU** compresses this same speed trajectory into a hidden state. After training on fast circuits, the hidden state has learned: *"If the speed trajectory has been climbing steeply, activate X-Mode prediction."* At Monaco, no sample looks like any training sample in terms of speed progression. GRU predicts Z-Mode for 99.5% of Monaco's test samples, yielding F1-Xmode = 0.005.

**LR-instant** has no memory. It evaluates: *"Is the current Speed approaching 240 km/h AND current nGear approaching 6?"* The approach to this threshold looks similar regardless of which circuit the car is on. Hence LR remains robust across all four circuits (σ = 0.021 vs GRU's σ = 0.104).

**Transformer** survives better than GRU because its attention mechanism focuses on instantaneous state features (Throttle, Speed, nGear — confirmed by Table 9) rather than accumulated dynamics. Attention is less sensitive to absolute speed magnitude than GRU's compressed trajectory representation.

### 5.2 Connection Between Phase 1 and Phase 2

Phase 1 established that the X-Mode threshold rule is recoverable from instantaneous features with perfect accuracy. Phase 2 shows that shifting this task to 1-second look-ahead on unseen circuits is hard precisely because models that encode circuit-specific patterns from training data fail.

This connection is direct: the anomalies detected by Isolation Forest in Phase 1 are braking events that deviate from the circuit's "normal" driving pattern (OR = 0.18 vs. X-Mode). GRU's hidden state represents the same circuit-specific "normal" speed profile. Monaco's lower speed envelope is to GRU what an anomalous braking event is to the Isolation Forest at a fast circuit — a statistical outlier from the learned distribution.

The models that survive cross-circuit evaluation (LR-instant, Transformer) are the models whose representations are not grounded in circuit-specific baselines. This is the unifying principle across both phases.

### 5.3 Practical Engineering Implications

1. **Control system design:** Real-time active aero systems should use H=10 (1-second) anticipation. The trivial regime (H≤5) provides no meaningful advance notice, and the breakdown regime (H≥25) is unreliable.

2. **Model selection:** For deployment on unseen circuits, LR-instant is the safest choice (lowest σ, highest mean AUC-ROC). If a deep model is required for other reasons, Transformer is the only deep architecture with demonstrated cross-circuit robustness.

3. **Feature design:** The most important features are instantaneous state (Throttle, Speed, nGear, RPM). Adding circuit-adaptive normalisation — expressing Speed as "fraction of circuit maximum" rather than absolute km/h — would likely close the gap between LR-instant and deep models.

---

## 6. Conclusions

This project demonstrates two connected findings:

**Phase 1:** The current aerodynamic mode is perfectly classifiable from instantaneous telemetry features (RF AUC-ROC = 1.000). This validates the feature set but establishes a ceiling, not a contribution. Anomalous driving windows (≈5% of data) exhibit large-effect deviations from normal patterns (Cohen's d up to −1.41), predominantly corresponding to braking events at technical corners.

**Phase 2:** Anticipatory prediction of aero mode 1 second ahead on unseen circuits is feasible but significantly harder. Logistic Regression (AUC-ROC = 0.963) is the most reliable model; no deep learning architecture surpasses it. The critical failure mode — circuit memorisation — affects every model that encodes absolute speed levels from training circuits. GRU collapses at Monaco (AUC-ROC = 0.728), while the Transformer maintains robustness (0.956) by focusing on instantaneous state features rather than calibrated speed dynamics.

**Summary answers to key research questions:**

| Question | Answer |
|---|---|
| Is current aero mode detectable? | Yes — RF AUC = 1.000 from instantaneous features |
| Are driver anomalies detectable? | Yes — 5% rate, large effect sizes, GPS-mapped to technical corners |
| Can we predict 1s ahead on unseen circuits? | Yes — AUC-ROC 0.963 (LR) to 0.818 (CNN) |
| Does deep learning improve prediction? | No — all deep models perform below LR-instant |
| Why do deep models fail? | Circuit memorisation of absolute speed levels |
| Which deep model is most robust? | Transformer (AUC 0.911, Monaco 0.956) |
| What is the useful horizon? | ≤ 1 second; all methods fail beyond 2.5 seconds |
| What are the most important features? | Throttle, Speed, nGear (instantaneous state) |
| What is the hardest test scenario? | Monaco street circuit + mode transition windows |

---

## 7. Future Work

**Addressing circuit memorisation (highest priority):**

1. **Circuit-adaptive normalisation:** Replace absolute Speed with speed relative to the circuit's mean or maximum (computable from training data without leaking test information). This would eliminate the calibration anchor that causes GRU and RF-lag to fail at Monaco and is the most direct solution to the identified root cause.

2. **Domain-adversarial training:** Add a circuit-classifier adversarial head with gradient reversal, forcing the encoder to learn circuit-invariant representations. This approach has proven effective in domain adaptation problems analogous to the circuit-shift problem here.

**Extending the evaluation:**

3. **Broader circuit coverage:** The current LOCO benchmark covers 4 circuits. Extending to a full 24-race season would provide 24-fold evaluation with sufficient statistical power for Wilcoxon tests (n≥10 required for p < 0.05 two-tailed).

4. **Multi-driver study:** Current data covers two drivers; behavioural differences are statistically significant but small (Cohen's d = 0.042). A full-grid, multi-season study would test whether driver-specific models provide meaningful benefit.

**Improving transition window performance:**

5. **Transition-aware loss function:** Up-weight the ±0.5-second windows around mode changes in the focal loss objective. Table 8 shows these are the most safety-critical prediction events and the current loss function treats them identically to stable windows.

**Deployment and outcome validation:**

6. **Edge deployment latency:** All Phase 2 models are small (68k–213k parameters). Benchmarking inference latency on embedded hardware representative of F1-grade ECUs would confirm real-time feasibility.

7. **Outcome-based label:** The current X-Mode label is a physics heuristic (fixed thresholds). A superior label would derive from lap-time simulations: *"would switching to X-Mode at this moment have improved lap time?"* This would align model objectives with engineering utility rather than rule recovery.

---

## 8. References

[1] FIA, "2026 Formula One Technical Regulations, Art. 3.10," Fédération Internationale de l'Automobile, 2024.

[2] T. Oehrly, "FastF1: A Python package for accessing and analysing Formula 1 data," GitHub, 2024.

[3] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," *Proc. IEEE ICCV*, pp. 2980–2988, 2017.

[4] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," *arXiv:1803.01271*, 2018.

[5] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all you need," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[6] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks (Integrated Gradients)," *Proc. 34th ICML*, pp. 3319–3328, 2017.

[7] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," *Proc. IEEE ICDM*, pp. 413–422, 2008.

[8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions (SHAP)," *NeurIPS*, 2017.

[9] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[10] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[11] E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988.

[12] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, "Deep learning for time series classification: a review," *Data Mining and Knowledge Discovery*, vol. 33, pp. 917–963, 2019.

[13] J. V. Casas and J. M. Vicen, "LSTM-based lap time prediction for Formula 1," *Journal of Sports Engineering and Technology*, 2022.

[14] W. J. Conover, *Practical Nonparametric Statistics*, 3rd ed. New York: Wiley, 1999.
