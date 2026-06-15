# Overall Project Report
## F1 Predictive AI and Driver Anomaly Analysis
### Phase 1: Anomaly Detection & Aero State Classification  
### Phase 2: Anticipatory Aerodynamic Mode Prediction

---

**Submitted by:** Anas Norani · Hanan Majeed · Maha Mohsin  
**Program:** BS Data Science — SEECS, NUST  
**Date:** June 2026  
**Conference Paper:** IEEE (submission deadline 21 June 2026)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Background and Motivation](#2-project-background-and-motivation)
3. [Dataset Overview](#3-dataset-overview)
4. [Phase 1 — Driver Anomaly Detection and Aero State Classification](#4-phase-1--driver-anomaly-detection-and-aero-state-classification)
   - 4.1 Objectives
   - 4.2 Data and Features
   - 4.3 Methods
   - 4.4 Results
   - 4.5 Statistical Analysis
   - 4.6 Key Findings and Limitations
5. [Phase 2 — Anticipatory Aero Prediction Across Circuits](#5-phase-2--anticipatory-aero-prediction-across-circuits)
   - 5.1 Objective and Problem Formulation
   - 5.2 Why It Is Harder Than Phase 1
   - 5.3 Data and Feature Set
   - 5.4 Evaluation Protocol (LOCO)
   - 5.5 Models
   - 5.6 Results
   - 5.7 Root Cause: Circuit Memorisation
   - 5.8 Interpretability
   - 5.9 Transition Window Analysis
   - 5.10 Key Findings
6. [Connecting the Phases](#6-connecting-the-phases)
7. [Complete Pipeline and Technical Details](#7-complete-pipeline-and-technical-details)
8. [Future Work](#8-future-work)
9. [Overall Conclusions](#9-overall-conclusions)
10. [References](#10-references)

---

## 1. Executive Summary

This project applies machine learning to Formula 1 car telemetry data across two phases.

**Phase 1** takes real telemetry from the 2026 Japanese Grand Prix at Suzuka and asks: can we detect unusual driving behaviour, cluster different driving regimes, and classify the car's current aerodynamic mode from sensor data? Using Isolation Forest, K-Means clustering, and classification models, we confirm that:
- Random Forest achieves **AUC-ROC = 1.000** classifying the current aero mode — the threshold rule is completely recoverable from instantaneous features.
- Isolation Forest identifies anomalous driving windows (primarily heavy-braking and gear-shift events) with **large effect sizes** (Cohen's d up to −1.41) against normal driving.
- Four distinct driving zones are statistically confirmed by Kruskal-Wallis (η² = 0.555).
- All features are non-normal (Shapiro-Wilk p < 0.001 for all), requiring non-parametric methods throughout.

**Phase 2** takes these insights and poses a harder question: can models predict the aero mode **1 second before** the car reaches it, generalising to circuits they have never seen? Using Leave-One-Circuit-Out evaluation across Monaco, Monza, Silverstone, and Suzuka with 8 models and 5 prediction horizons:
- **LR-instant achieves the best overall AUC-ROC of 0.963**, beating all deep learning models.
- **GRU catastrophically collapses at Monaco** (AUC-ROC = 0.728, F1-xmode = 0.005).
- **Transformer is the most circuit-robust deep model** (AUC-ROC = 0.911, Monaco = 0.956).
- All models fail beyond a 2.5-second horizon. Practical systems must operate within ~1 second.
- The root cause of deep model failure is **circuit memorisation** — models encode absolute speed levels from training circuits, which do not transfer to Monaco's lower speed envelope.

The core insight connecting both phases: Phase 1 shows that the current aero state is trivially detectable from instantaneous speed and gear features. Phase 2 shows that extending this to 1-second look-ahead across different circuits is genuinely hard — and that models which "memorise" circuit-specific dynamics (the same pattern Isolation Forest flagged as anomalous in Phase 1) are exactly the ones that fail at unseen circuits.

---

## 2. Project Background and Motivation

Formula 1 cars in the 2026 technical regulations (FIA Art. 3.10) use **Active Aerodynamics** — a system that physically changes wing geometry depending on driving conditions:

- **Z-Mode (High Downforce):** Wings are fully deployed for maximum grip in slow corners.
- **X-Mode (Low Drag):** Wings flatten to reduce air resistance on fast straights.

The switch is currently triggered reactively: the car measures speed ≥ 240 km/h AND gear ≥ 6, then actuates the wing. This creates a lag — by the time the sensor fires, the car has already entered the straight and the optimal switching moment may have passed.

The engineering question is: can a control system **anticipate** that the car is about to enter X-Mode conditions and switch early? This requires predicting a future state from present sensor readings.

This project attacks that question in two steps:

1. **First, understand the current state.** Before predicting the future, verify that the current aero state is detectable from telemetry, characterise how the car behaves differently in different modes, and identify any unusual or anomalous driving patterns. This is Phase 1.

2. **Then, predict the future.** Using the feature understanding from Phase 1, build and evaluate models that predict the aero mode 1 second ahead on circuits they have never trained on. This is Phase 2.

---

## 3. Dataset Overview

All data was collected using the open-source **FastF1** Python library from official FIA timing feeds at approximately 10 Hz (10 samples per second per car).

### Phase 1 Dataset

| Item | Value |
|---|---|
| Race | 2026 Japanese Grand Prix |
| Circuit | Suzuka International Racing Course |
| Drivers | Max Verstappen (Red Bull) · Isack Hadjar (Red Bull) |
| Total rows | 63,673 |
| Sampling rate | ~10 Hz |
| X-Mode rate | 35% |

### Phase 2 Dataset

| Circuit | Race | Driver(s) | Rows | X-Mode Rate |
|---|---|---|---|---|
| Monaco | 2025 Monaco GP | Verstappen | 39,881 | 13% |
| Monza | 2025 Italian GP | Verstappen | 31,098 | 57% |
| Silverstone | 2025 British GP | Verstappen | 25,060 | 38% |
| Suzuka | 2026 Japanese GP | Verstappen + Hadjar | 63,673 | 35% |
| **Total** | | | **159,712** | **~36% overall** |

The four circuits represent maximally diverse conditions: Monaco is a slow street circuit (top speed ~220 km/h, only 13% X-Mode); Monza is a pure power circuit (top speed ~365 km/h, 57% X-Mode). This diversity intentionally stress-tests generalisation.

### Target Label

The aero mode label is derived from the physical activation rule:

```
y = 1 (X-Mode)  if  Speed[t+H] ≥ 240.0 km/h  AND  nGear[t+H] ≥ 6
y = 0 (Z-Mode)  otherwise
```

The critical detail: the label uses the value at time **t + H** (the future), not the current time step. The model sees only data up to time **t**.

---

## 4. Phase 1 — Driver Anomaly Detection and Aero State Classification

### 4.1 Objectives

Phase 1 has three goals:
1. Detect anomalous driving behaviour in Suzuka 2026 telemetry using unsupervised learning.
2. Cluster the telemetry into meaningful driving zones.
3. Classify the current aerodynamic mode from instantaneous and engineered features.
4. Run rigorous statistical tests to characterise distributions, driver differences, and feature importance.

### 4.2 Data and Features

**Raw features (8):** Speed, RPM, nGear, Throttle, Brake, Acceleration, Elevation_Delta, Tire_Age_Laps

**Engineered features (10):**

| Feature | Formula / Definition |
|---|---|
| Kinetic_Energy_MJ | ½ × m × v² (m ≈ 800 kg), converted to MJ |
| Longitudinal_Force_N | m × Acceleration |
| High_Speed_Zone | Binary: Speed ≥ 240 AND nGear ≥ 6 |
| Heavy_Braking | Binary: Brake ≥ 50% AND Acceleration ≤ −5 m/s² |
| Gear_Shift_Active | Binary: consecutive gear changes within 0.5s window |
| Speed_Rolling_Avg | Rolling mean of Speed over 10 samples (1s) |
| Track_Zone | K-Means cluster label (k=4) |
| Compound_Encoded | Tyre compound as integer (Soft=0, Medium=1, Hard=2) |

Note: Engine_Load (= RPM × Throttle / 100) and Energy_Efficiency_Ratio were also present in Phase 1 artefacts but are fabricated proxies not used in Phase 2 to maintain feature integrity.

**Descriptive statistics (key features, n=63,673):**

| Feature | Mean | Std | Min | Max | Skewness |
|---|---|---|---|---|---|
| Speed (km/h) | 218.2 | 65.4 | 58.0 | 347.0 | −0.21 |
| RPM | 10,619 | 811 | 6,252 | 12,708 | −0.96 |
| nGear | 5.48 | 1.90 | 1 | 8 | −0.40 |
| Throttle (%) | 62.5 | 42.5 | 0 | 104 | −0.47 |
| Acceleration (m/s²) | 0.01 | 4.85 | −28.0 | 28.0 | −0.81 |
| Elevation_Delta (m) | 0.00 | 4.34 | −82.0 | 108.5 | −0.07 |
| Kinetic_Energy (MJ) | 1.60 | 0.86 | 0.10 | 3.71 | 0.31 |

### 4.3 Methods

**Isolation Forest (anomaly detection)**
- Contamination parameter: 5% (estimated from visual inspection of lap data)
- 100 estimators, max_samples='auto', random_state=42
- Anomalies scored by average path length across all trees

**K-Means clustering (driving zone segmentation)**
- k=4 determined by elbow method on inertia curve
- Features used: Speed, nGear, Acceleration, Brake, Throttle (normalised)
- Cluster assignment used as `Track_Zone` engineered feature

**Classification models**
- **Logistic Regression:** L2 regularisation (C=1.0), solver='lbfgs', max_iter=1000
- **Random Forest:** 100 trees, max_depth=None, min_samples_split=2, random_state=42
- Train/test split: 80/20 stratified by aero mode label
- Evaluation: AUC-ROC, F1-Score, Precision, Recall, bootstrap CI (10,000 resamples)

### 4.4 Results

#### 4.4.1 Anomaly Detection

Isolation Forest flagged approximately **5% of rows** as anomalous. Statistical comparison of anomalous vs normal windows:

| Feature | Anomaly Mean | Normal Mean | Cohen's d | Effect Size |
|---|---|---|---|---|
| Acceleration (m/s²) | −6.20 | +0.34 | −1.41 | **Large** |
| Throttle (%) | 22.7 | 64.6 | −1.01 | **Large** |
| Brake | 0.686 | 0.157 | +1.44 | **Large** |
| Gear_Shift_Active | 0.711 | 0.022 | +3.96 | **Large** |
| Heavy_Braking | 0.290 | 0.000 | +2.86 | **Large** |

All differences are statistically significant (Mann-Whitney U, p < 0.001 for all features). The anomaly signature is consistent: **maximum braking, near-zero throttle, active gear shifting, and strong negative acceleration** — corresponding to late braking into corners and traction limit events.

Association with aero state: Chi-squared test shows anomalies are strongly non-independent from aero mode (χ² = 817.4, p = 9.0×10⁻¹⁸⁰, OR = 0.18 [CI: 0.16–0.20]). Anomalous windows are only 18% as likely to be in X-Mode as normal windows — confirming that anomalies predominantly occur during Z-Mode braking events, not on straights.

When mapped to GPS coordinates, anomalies cluster at Suzuka's known technical sections: the Esses complex, Degner curves, the Hairpin (Turn 11), Spoon Curve, and the 130R approach.

#### 4.4.2 Cluster Analysis

K-Means (k=4) separates the telemetry into four statistically distinct zones (Kruskal-Wallis on Speed: H = 34,439, p < 0.001, η² = 0.555 — large effect, cluster membership explains 55.5% of Speed variance):

| Cluster | Dominant Behaviour | Typical Speed | Aero Mode |
|---|---|---|---|
| 0 | High-speed aero phase | ≥ 240 km/h | X-Mode |
| 1 | Medium-speed cornering | 180–240 km/h | Z-Mode |
| 2 | Heavy braking and entry | < 180 km/h | Z-Mode |
| 3 | Slow hairpin / chicane | < 130 km/h | Z-Mode |

Post-hoc Dunn's test confirms all pairs of clusters are statistically distinct (p < 0.001 for all 6 pairwise comparisons after Bonferroni correction).

#### 4.4.3 Classification Results

Both classifiers are evaluated on the held-out test split (20% of 63,673 rows = ~12,735 samples):

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Logistic Regression | 0.979 | 0.881 | 1.000 | 0.937 | 0.9997 |

Bootstrap confidence intervals (10,000 resamples):

| Model | AUC-ROC | 95% CI |
|---|---|---|
| Random Forest | 1.000 | [1.000, 1.000] |
| Logistic Regression | 0.9997 | [0.9993, 0.9999] |

DeLong AUC test: ΔAUC = 0.0003, z = −2.06, p = 0.039 — the difference is statistically significant, though the practical gap is negligible (both are essentially perfect). McNemar test shows RF wins on 274 edge-case instances vs 0 for LR (χ² = 272, p ≈ 0), confirming RF handles the non-linear speed-gear interaction boundary slightly better.

**Interpretation:** Both models achieve near-perfect performance because the label is defined by thresholds on features that are present in the input. This is expected — it validates the feature set but provides no evidence of predictive power for future states. This is the direct motivation for Phase 2.

### 4.5 Statistical Analysis

#### 4.5.1 Normality Testing

All 11 features were tested for normality using both Shapiro-Wilk (on a 5,000-sample subsample) and D'Agostino-Pearson K².

**Result: All 11 features are non-normal (p < 0.001 for both tests on all features).** Speed shows slight bimodality (two peaks corresponding to aero modes). Throttle is heavily bimodal (0% and 100% dominant). Acceleration has heavy tails (excess kurtosis = 10.0). Elevation_Delta has extreme kurtosis (excess kurtosis = 95.0) from the Dunlop and Degner altitude changes.

Non-parametric tests are used throughout subsequent analyses.

#### 4.5.2 Driver Comparison (Verstappen vs Hadjar)

| Test | Result | p-value | Effect |
|---|---|---|---|
| Mann-Whitney U on Speed | U = 0.0 | < 0.001 | d = 0.042 (small) |
| Chi-squared on aero deviation rate | χ² = 73.1 | < 0.001 | Cramér's V = 0.113 (small) |

Both differences are statistically significant but practically small (Cohen's d < 0.1, Cramér's V < 0.2). Verstappen shows marginally higher average speed and a slightly different aero deviation rate. As a rookie in only his second full season, Hadjar's more conservative cornering entry is consistent with the small but real driver signature.

#### 4.5.3 Track Zone Statistical Distinctness

Kruskal-Wallis test on Speed across 4 K-Means clusters: **H = 34,439, p < 0.001, η² = 0.555**. Cluster membership explains 55.5% of Speed variance — confirming the clusters are physically meaningful, not artefacts. All 6 pairwise Dunn comparisons significant at p < 0.001 (Bonferroni corrected).

#### 4.5.4 Permutation Feature Importance

| Rank | Feature | Importance | 95% CI | p-value |
|---|---|---|---|---|
| 1 | nGear | 0.167 | [0.155, 0.180] | < 0.001 |
| 2 | Elevation_Delta | 0.162 | [0.156, 0.170] | < 0.001 |
| 3 | Brake | 0.081 | [0.075, 0.086] | < 0.001 |
| 4 | Speed | 0.0004 | [0.000, 0.0005] | < 0.001 |

The dominance of **nGear over Speed** in permutation importance is striking. It reflects that nGear is the harder constraint to satisfy for X-Mode (high gear requires both high speed AND appropriate track position), while Speed alone is insufficient. Elevation_Delta's high importance links to Suzuka's distinctive terrain: elevation changes reliably co-occur with specific speed zones.

#### 4.5.5 Bootstrap Confidence Intervals

Mean Speed in X-Mode: 291.4 km/h [CI: 290.8, 292.1]. Mean Speed in Z-Mode: 183.9 km/h [CI: 183.4, 184.4]. Confidence intervals are non-overlapping by > 100 km/h — the speed separation between modes is unambiguous and robust.

### 4.6 Key Findings and Limitations

**Key findings:**
1. The current aero mode is perfectly classifiable from instantaneous features (RF AUC = 1.000). This is a necessary sanity check, not a contribution.
2. Anomalous driving windows are predominantly braking events (OR = 0.18 vs X-Mode) and are geographically clustered at technical sections of Suzuka.
3. Four distinct driving zones exist with large statistical separation (η² = 0.555).
4. nGear and Elevation_Delta are the most important features for current-state classification (permutation importance), not raw Speed.
5. Both drivers show statistically different but practically similar behaviours (small effect sizes).

**Critical limitation:** Both the anomaly detector and classifiers are trained and evaluated on **the same circuit and session**. There is no test of whether these patterns hold on other circuits. This is the direct motivation for Phase 2.

---

## 5. Phase 2 — Anticipatory Aero Prediction Across Circuits

### 5.1 Objective and Problem Formulation

Phase 2 formalises the anticipatory prediction task:

Given a sliding window of the last **W = 50** time steps (5 seconds at 10 Hz) ending at time **t**, predict the aero mode label at time **t + H**.

```
Input:  X = [x_{t-W+1}, x_{t-W+2}, ..., x_t]   shape: (50, 12)
Target: y = 1(Speed[t+H] ≥ 240) AND 1(nGear[t+H] ≥ 6)
```

The label index is strictly greater than the last input index. No future information leaks into the input. This is a **binary time-series classification** task.

Primary horizon: **H = 10** (1 second ahead). Four additional horizons are evaluated for the sweep: H=1 (0.1s), H=5 (0.5s), H=25 (2.5s), H=50 (5s).

### 5.2 Why It Is Harder Than Phase 1

Phase 1 achieved AUC = 1.000 on the current aero mode. Phase 2 introduces three additional challenges:

1. **Future shift.** The target is the aero mode at *t+H*, but current Speed and nGear (the defining features) are not available to the model. It must infer from the trajectory whether the car is *approaching* the threshold.

2. **Cross-circuit generalisation.** Models train on 3 circuits and test on the 4th. The speed envelope varies dramatically: Monaco's top speed (~220 km/h) is 40% lower than Monza's (~365 km/h). A model that memorises "X-Mode typically happens at 280+ km/h" will fail at Monaco, where the maximum speed rarely exceeds 220 km/h but X-Mode still activates at the correct threshold.

3. **Class imbalance.** X-Mode rates range from 13% (Monaco) to 57% (Monza). A na baseline always predicts Z-Mode, achieving up to 87% accuracy at Monaco while being completely useless.

### 5.3 Data and Feature Set

**Feature set (F = 12 — consistent across both phases' core features):**

| Feature | Category |
|---|---|
| Speed | Raw telemetry |
| RPM | Raw telemetry |
| nGear | Raw telemetry |
| Throttle | Raw telemetry |
| Brake | Raw telemetry |
| X, Y, Z | GPS position |
| Acceleration | Derived (Δspeed/Δtime) |
| Elevation_Delta | Derived (ΔZ/Δtime) |
| Kinetic_Energy_MJ | Physics (½mv²) |
| Longitudinal_Force_N | Physics (m × Acceleration) |

Each window is shape (50, 12). StandardScaler is fit on training windows only and applied to test windows — no data leakage.

### 5.4 Evaluation Protocol (LOCO)

**Leave-One-Circuit-Out (LOCO):**

| Fold | Train | Test |
|---|---|---|
| 1 | Monza + Silverstone + Suzuka | Monaco |
| 2 | Monaco + Silverstone + Suzuka | Monza |
| 3 | Monaco + Monza + Suzuka | Silverstone |
| 4 | Monaco + Monza + Silverstone | Suzuka |

Four folds produce four per-model AUC-ROC values. Final reported statistics are the mean ± std (ddof=1) across these four fold-level observations.

LOCO is substantially harder than random train/test splits: any model that memorises circuit-specific features will fail on the unseen fold. A model that achieves consistent AUC across all four folds has genuinely learned transferable physics.

**Class imbalance handling:** Focal Loss (γ = 2.0, α = 1 − pos_rate) for all deep models. Classical models use class_weight='balanced'.

### 5.5 Models

**Classical — no temporal reasoning:**

| Model | Description | Parameters |
|---|---|---|
| LR-instant | Logistic Regression on 1 × 12 instantaneous features | ~12 |
| RF-instant | Random Forest on 1 × 12 instantaneous features | ~50k (100 trees) |

**Classical — temporal (weak):**

| Model | Description | Parameters |
|---|---|---|
| RF-lag | Random Forest on flattened 50×12 = 600-dim input | ~50k |

**Deep Learning — temporal:**

| Model | Architecture | Parameters |
|---|---|---|
| CNN1D | Two temporal conv layers + dense head | ~136k |
| CausalLSTM | 2-layer LSTM with causal masking | ~213k |
| CausalGRU | 2-layer GRU with causal masking | ~162k |
| TCN | Dilated causal convolutions (4 layers, dilation 1,2,4,8) | ~152k |
| CausalTransformer | 2-layer encoder with causal mask + last-token classification | ~68k |

All deep models trained with AdamW (lr=3×10⁻⁴, weight_decay=1×10⁻⁴), early stopping (patience=8), cosine annealing LR schedule, and automatic mixed precision (AMP) on NVIDIA P100 GPU (Kaggle free tier).

### 5.6 Results

#### 5.6.1 Primary Results — H=10 (1 Second Ahead)

Mean ± std across 4 LOCO folds (ddof=1):

| Model | F1-Macro | AUC-ROC | AUC-PR | σ (AUC-ROC) |
|---|---|---|---|---|
| **LR-instant** | **0.906 ± 0.022** | **0.963 ± 0.021** | **0.881 ± 0.067** | **0.021** |
| RF-instant | 0.872 ± 0.050 | 0.959 ± 0.011 | 0.864 ± 0.099 | 0.011 |
| RF-lag | 0.794 ± 0.148 | 0.945 ± 0.026 | 0.815 ± 0.162 | 0.026 |
| **Transformer** | 0.779 ± 0.115 | **0.911 ± 0.055** | 0.783 ± 0.096 | 0.055 |
| LSTM | 0.756 ± 0.139 | 0.878 ± 0.085 | 0.729 ± 0.121 | 0.085 |
| GRU | 0.768 ± 0.179 | 0.877 ± 0.104 | 0.724 ± 0.273 | 0.104 |
| TCN | 0.785 ± 0.117 | 0.835 ± 0.201 | 0.755 ± 0.167 | 0.201 |
| CNN | 0.748 ± 0.093 | 0.818 ± 0.111 | 0.712 ± 0.148 | 0.111 |

The performance ranking from Phase 1 (RF ≈ LR >> deep models) inverts in magnitude: in Phase 1, all models were near-perfect (0.9997–1.000). In Phase 2, there is now real separation between models, confirming that the task is genuinely harder.

#### 5.6.2 Per-Circuit LOCO Breakdown

AUC-ROC at each held-out circuit:

| Model | Monaco | Monza | Silverstone | Suzuka |
|---|---|---|---|---|
| LR-instant | **0.980** | 0.932 | 0.970 | 0.969 |
| RF-instant | 0.955 | **0.963** | **0.972** | 0.945 |
| RF-lag | 0.907 | 0.947 | 0.963 | **0.961** |
| **Transformer** | **0.956** | 0.871 | 0.961 | 0.857 |
| LSTM | 0.927 | 0.863 | 0.957 | 0.765 |
| **GRU** | **0.728** ← collapse | 0.896 | 0.968 | 0.917 |
| TCN | 0.944 | 0.890 | 0.968 | **0.537** ← collapse |
| CNN | 0.811 | 0.821 | 0.956 | 0.684 |

Two catastrophic failures are highlighted:
- **GRU at Monaco: AUC-ROC = 0.728, F1-xmode = 0.005.** The model predicts Z-Mode for nearly every sample. On the same circuit and same data, Transformer achieves 0.956.
- **TCN at Suzuka: AUC-ROC = 0.537** — essentially random guessing.

#### 5.6.3 Horizon Sweep

Mean AUC-ROC across 4 LOCO folds at each prediction horizon (selected models):

| Model | H=1 (0.1s) | H=5 (0.5s) | H=10 (1s) | H=25 (2.5s) | H=50 (5s) |
|---|---|---|---|---|---|
| LR-instant | 0.999 | 0.993 | 0.963 | 0.786 | 0.471 |
| RF-instant | 0.999 | 0.993 | 0.959 | 0.777 | 0.562 |
| RF-lag | 0.999 | 0.985 | 0.945 | 0.745 | 0.466 |
| Transformer | 0.999 | 0.987 | 0.911 | 0.691 | **0.595** |
| GRU | 0.999 | 0.986 | 0.877 | 0.682 | 0.505 |

At H=1 and H=5: all models are near-perfect — the car barely changes state in 0.1–0.5 seconds, making the task trivially easy. At H=10: real differences emerge; this is the operationally relevant horizon. At H=25+: all models degrade sharply. At H=50: LR collapses to AUC = 0.471 (below chance), while Transformer retains 0.595 — the only model above chance at 5 seconds.

**Practical implication for system design:** A real active aero system should use a prediction horizon of approximately 0.5–1.0 seconds. Beyond 2.5 seconds, no current method is reliable enough for a production system.

### 5.7 Root Cause: Circuit Memorisation

The most counter-intuitive finding is that the simplest model (LR-instant, 12 parameters) beats all deep models (162k–213k parameters) under LOCO evaluation. The explanation is **circuit memorisation**.

**How RF-lag memorises circuits:**  
RF-lag sees the full 5-second speed history. It learns: *"When speed has been 250–280 km/h for the last 5 seconds and is still climbing, we are approaching a fast straight."* This rule is 94.5% accurate at Monza (where top speed exceeds 360 km/h and the pattern fires correctly) and 96.3% accurate at Silverstone. At Monaco, the entire circuit is slower — maximum speed rarely exceeds 220 km/h. The "fast straight approach" signal never fires, and RF-lag's Monaco AUC drops to 0.907. Still acceptable, but the gap vs LR is already visible.

**How GRU memorises circuits more severely:**  
GRU compresses the speed trajectory into a fixed-size hidden state. After training on fast circuits, its hidden state encodes: *"am I in a regime where fast straights occur?"* At Monaco, no sample in the test fold looks like any training sample in terms of speed trajectory. GRU's hidden state never activates its "X-Mode incoming" representation. The result: **F1-xmode = 0.005** at Monaco — the model predicts X-Mode on only 0.5% of samples where it is actually required.

**Why LR is immune:**  
LR has no memory. It evaluates: *"Is the current Speed ≥ ~220 km/h AND current nGear ≥ ~5?"* These thresholds generalise trivially across circuits because the absolute activation rule (Speed ≥ 240, nGear ≥ 6) is the same at Monaco as at Monza.

**Why Transformer is more robust than GRU:**  
Transformer's attention mechanism focuses primarily on the most recent 5–10 time steps (confirmed by attention maps). It effectively behaves like a sophisticated instantaneous estimator rather than accumulating circuit-calibrated trajectory patterns. Its top-ranked features at Monaco are Throttle, Speed, and nGear — the same instantaneous state features that make LR robust.

This finding is directly connected to Phase 1: Isolation Forest detected anomalies based on deviations from circuit-specific driving patterns. Models that depend on such circuit-specific patterns for prediction are exactly the models that fail under LOCO evaluation.

### 5.8 Interpretability

#### 5.8.1 Integrated Gradients (IG)

IG attributions are computed on 300 test windows at Monaco, comparing GRU and Transformer:

| Feature | GRU Importance | Transformer Importance |
|---|---|---|
| Longitudinal Force | **#1 (9.9%)** | #5 (3.7%) |
| Acceleration | **#2 (9.7%)** | #6 (3.3%) |
| Throttle | #3 (8.5%) | **#1 (12.0%)** |
| Speed | #4 (8.1%) | **#2 (10.2%)** |
| nGear | #8 | **#3 (9.6%)** |
| RPM | #9 | **#4 (7.8%)** |

GRU's top features (Force, Acceleration) are **rate-of-change dynamics** — signals that tell the model how fast the car is changing state. At fast circuits, a steep acceleration trajectory reliably precedes X-Mode activation. At Monaco, the same dynamics never reach the trained thresholds.

Transformer's top features (Throttle, Speed, nGear) are **instantaneous state** — what the car is doing right now. These transfer directly: if Throttle = 100% AND Speed = 235 AND nGear = 5, an X-Mode transition is imminent regardless of which circuit the car is on.

#### 5.8.2 Transformer Attention Maps

Average self-attention weights (50×50) across all test samples and encoder layers show consistent topology across all four circuits: strong attention on the most recent 5–10 time steps, with a secondary signal at the very first time step (long-range context). This pattern does not reorganise between Monaco and Monza — confirming that Transformer's robustness stems from consistent attention behaviour rather than circuit-specific adaptation.

### 5.9 Transition Window Analysis

Mode changes are sparse, high-value events. We analysed prediction quality within ±5 samples (±0.5s) of each actual mode transition versus all other (stable) windows.

| Circuit | Model | F1 (Transitions) | F1 (Stable) | Drop |
|---|---|---|---|---|
| Monaco | Transformer | 0.496 | 0.902 | −0.406 |
| Monaco | GRU | 0.328 | 0.480 | −0.152 |
| Monza | Transformer | 0.551 | 0.865 | −0.314 |
| Monza | GRU | 0.615 | 0.943 | −0.327 |
| Silverstone | Transformer | 0.659 | 0.970 | −0.311 |
| Silverstone | GRU | 0.674 | 0.973 | −0.299 |
| Suzuka | Transformer | 0.554 | 0.599 | −0.045 |
| Suzuka | GRU | 0.575 | 0.860 | −0.285 |
| **MEAN** | **Transformer** | **0.565** | **0.834** | **−0.269** |
| **MEAN** | **GRU** | **0.548** | **0.814** | **−0.266** |

The ~0.27 F1 drop at transition windows is consistent across both models and all circuits. This is a fundamental task property: the model is asked to predict a regime change from telemetry that has not yet fully entered the new regime. GRU's Monaco transition F1 (0.328) is particularly low — it barely predicts correctly even at the moment of change, because its stable-state prediction is already wrong.

Future work should incorporate a **transition-aware loss function** that up-weights the ±0.5s windows around mode changes, penalising the most safety-critical prediction errors more heavily.

### 5.10 Key Findings

1. **LR-instant achieves the best overall AUC-ROC of 0.963** — simpler is better under cross-circuit evaluation.
2. **Deep learning does not help here.** Complex temporal models memorise circuit-specific patterns and fail on unseen circuits.
3. **The Transformer is the most circuit-robust deep model** (AUC-ROC 0.911, Monaco 0.956) because it focuses on instantaneous state rather than calibrated dynamics.
4. **GRU catastrophically fails at Monaco** (AUC-ROC 0.728, F1-xmode 0.005) — the most severe failure mode in the study.
5. **Useful prediction is only possible within ~1 second.** All models fail beyond 2.5 seconds.
6. **Transition windows are the hardest samples** — ~0.27 mean F1 drop at mode changes for both top models.

---

## 6. Connecting the Phases

The two phases form a coherent narrative:

| Question | Phase 1 Answer | Phase 2 Implication |
|---|---|---|
| Is current aero state detectable? | Yes — RF AUC = 1.000 | The task is solvable; now test if future state is predictable |
| What features define aero mode? | nGear and Speed are the defining threshold features | Models should focus on instantaneous state, not dynamics |
| Are there unusual driving patterns? | Yes — 5% anomaly rate (braking events, gear shifts) | Models that memorise "normal" circuit speed profiles will fail on anomalous circuits (Monaco) |
| Do drivers differ? | Yes, significantly but weakly (d < 0.1) | Driver-specific models are not worth the complexity for this task |
| Is current circuit data sufficient? | No — same-circuit train/test is trivially easy | LOCO is required for honest evaluation |

The deepest connection: Phase 1's anomaly detector and Phase 2's GRU fail for the same structural reason. Isolation Forest flags points that deviate from the circuit-specific "normal" speed distribution. GRU's hidden state encodes the same circuit-specific normal. Monaco's lower speed envelope is "anomalous" by Phase 1's Isolation Forest standards when measured against Monza's training data — and GRU's hidden state treats it identically.

The models that survive LOCO (LR, Transformer) are the models that do not depend on circuit-specific baselines.

---

## 7. Complete Pipeline and Technical Details

### Phase 1 Pipeline

| Script / Notebook | Purpose |
|---|---|
| `anomaly_detection/notebooks/01_Data_Preprocessing.ipynb` | FastF1 download, raw telemetry cleaning, feature engineering, train/test split |
| `anomaly_detection/notebooks/02_EDA.ipynb` | Exploratory visualisations, distribution plots, track maps |
| `anomaly_detection/notebooks/03_Modelling.ipynb` | Isolation Forest, K-Means, LR, RF training and evaluation |
| `anomaly_detection/notebooks/04_Statistical_Analysis.ipynb` | Normality tests, driver comparison, bootstrap CI, permutation importance, master dashboard |
| `anomaly_detection/app.py` | Streamlit app prototype (interactive anomaly explorer) |

**Outputs:** `anomaly_detection/artefacts/` (12 files including processed CSVs and trained models), `anomaly_detection/stats_outputs/` (11 statistical summary CSVs), `anomaly_detection/graphs/` (44 visualisation PNGs).

### Phase 2 Pipeline

| Script | Purpose |
|---|---|
| `anticipatory_aero/multi_circuit_work/01_build_anticipatory_dataset.py` | Downloads multi-circuit telemetry, engineers 12 features, builds sliding windows (W=50), saves processed numpy arrays |
| `anticipatory_aero/multi_circuit_work/02_baselines.py` | Trains LR-instant, RF-instant, RF-lag across all circuits and all 5 horizons |
| `anticipatory_aero/multi_circuit_work/03_deep_models.py` | GPU training of CNN, LSTM, GRU, TCN, Transformer with focal loss, early stopping, AMP, checkpointing |
| `anticipatory_aero/multi_circuit_work/04_evaluate.py` | Horizon sweep tables, bootstrap CIs, LaTeX table output, LOCO summary |
| `anticipatory_aero/multi_circuit_work/05_xai.py` | Integrated Gradients heatmaps, Transformer attention maps, SHAP for RF-lag |
| `anticipatory_aero/multi_circuit_work/trans_f1.py` | Low-memory transition window F1 analysis (per-circuit, explicit GC) |
| `anticipatory_aero/multi_circuit_work/generate_plots.py` | Generates 7 summary publication-quality plots using results CSVs |

**Training infrastructure:** Kaggle free GPU tier (NVIDIA P100, 16 GB VRAM). Total GPU time ≈ 6–8 hours for all 52 deep model training runs. All results are reproducible with `seed = 42`.

**Key outputs:**
- `anticipatory_aero/Results/baseline_results.csv` — 60 rows (3 models × 5 horizons × 4 circuits)
- `anticipatory_aero/Results/deep_results.csv` — 20 rows (5 deep models × 4 circuits, H=10)
- `anticipatory_aero/Results/sweep_results.csv` — 40 rows (GRU + Transformer × 5 horizons × 4 circuits)
- `anticipatory_aero/multi_circuit_work/graphs/` — 19 result PNGs

---

## 8. Future Work

### Building on Phase 1

1. **Cross-circuit anomaly detection:** Apply Phase 1's Isolation Forest trained on Suzuka to Monaco and Monza data. Identify whether the anomalous driving events are circuit-universal (braking at technical sections everywhere) or circuit-specific.

2. **Driver generalisation:** Phase 1 shows Verstappen and Hadjar have statistically different profiles. A multi-driver study across a full season would reveal whether driver-specific anomaly signatures exist.

3. **Outcome-linked labels:** Replace the rule-based anomaly flag with a lap-time-derived metric — *"did the driver lose more than X ms at this corner compared to their ideal lap?"* — to detect performance-relevant anomalies rather than just statistical outliers.

### Building on Phase 2

4. **Circuit-adaptive normalisation:** Replace absolute Speed with speed relative to the circuit-specific mean (computed on training circuits). This directly addresses the circuit-memorisation root cause identified in Section 5.7 and would likely close the Transformer-to-LR gap.

5. **Domain-adversarial training:** Add a circuit-classifier auxiliary head trained adversarially (gradient reversal). Forces the main encoder to learn circuit-invariant features.

6. **Transition-aware loss:** Up-weight the ±0.5s windows around each mode change in the focal loss. Phase 2's transition analysis (Section 5.9) shows this is the highest-value subproblem — a model that is correct at transitions would be far more useful for real actuation timing.

7. **Extend to more circuits:** Four circuits is a small LOCO test bed. A full 2025 season (24 races) would provide 24-fold LOCO evaluation and much stronger statistical power (current n=4 folds prohibits Wilcoxon tests from reaching p < 0.125).

8. **Edge deployment latency study:** All Phase 2 models are small (68k–213k parameters). Run inference latency benchmarks on embedded hardware representative of F1-grade ECUs to confirm real-time feasibility.

---

## 9. Overall Conclusions

| Research Question | Answer |
|---|---|
| Is current aero state detectable from telemetry? | Yes — RF AUC = 1.000. Trivially solvable from instantaneous Speed + nGear. |
| Are there detectable anomalies in F1 driving data? | Yes — ~5% anomaly rate; braking/gear-shift signature; large effect sizes (d up to −1.41). |
| Can we predict aero mode 1 second ahead on unseen circuits? | Yes — AUC-ROC 0.963 (LR) to 0.818 (CNN) across 4 LOCO folds. |
| Does deep learning outperform simple models for this task? | No — LR-instant beats all deep learning models under LOCO evaluation. |
| Why do complex temporal models fail? | Circuit memorisation: they encode absolute speed levels from training circuits. |
| Which deep model is most robust? | Transformer (AUC-ROC 0.911, Monaco 0.956) — focuses on instantaneous state. |
| What is the hardest generalisation scenario? | Monaco street circuit (13% X-Mode, low speed envelope). GRU collapses to AUC 0.728. |
| What is the useful prediction horizon? | ≤ 1 second. All models fail meaningfully beyond 2.5 seconds. |
| What features drive robust predictions? | Throttle, Speed, nGear (instantaneous state) — not dynamics or speed trajectory. |
| What is the single biggest open problem? | Circuit-adaptive normalisation to remove the speed-scale anchor from temporal models. |

**Summary statement:** This project demonstrates that anticipatory aero mode prediction at 1-second horizon is feasible and practically useful, but the dominant challenge is cross-circuit generalisation rather than temporal sequence modelling. The feature representations that explain Phase 1's anomaly detection (circuit-specific speed baselines) are exactly the representations that cause deep models to fail in Phase 2. Solving generalisation — likely through circuit-adaptive normalisation or adversarial domain adaptation — is the single most impactful next step.

---

## 10. References

1. FIA 2026 Formula 1 Technical Regulations, Article 3.10 — Active Aerodynamic Systems. Fédération Internationale de l'Automobile, 2025.
2. Oehrly, T. FastF1: A Python package for accessing and analysing Formula 1 data. GitHub (2024). https://github.com/theOehrly/Fast-F1
3. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.
4. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
5. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
6. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 3319–3328.
7. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE International Conference on Data Mining (ICDM)*, 413–422.
8. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics*, 1, 281–297.
9. Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
10. DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics*, 44(3), 837–845.
