# 🏎️ Aero Intelligence
### Predictive Machine Learning for Active Aerodynamics Efficiency and Driver Anomaly Analysis in Formula 1

> **CS-245 Machine Learning — Final Project Report**
> National University of Sciences & Technology (NUST)

| Member | CMS ID |
|---|---|
| Hanan Majeed | 519166 |
| Anas Norani | 501231 |
| Maha Mohsin | 504285 |

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Background & Motivation](#-background--motivation)
3. [Dataset](#-dataset)
4. [Pipeline Architecture](#-pipeline-architecture)
5. [Feature Engineering](#-feature-engineering)
6. [Models](#-models)
7. [Results](#-results)
8. [Statistical Validation](#-statistical-validation)
9. [Key Findings](#-key-findings)
10. [Limitations & Future Work](#-limitations--future-work)
11. [Repository Structure](#-repository-structure)
12. [References](#-references)

---

## 🔍 Project Overview

**Aero Intelligence** is a four-phase machine learning pipeline applied to 10 Hz F1 telemetry from the **2026 Japanese Grand Prix at Suzuka**. It addresses a brand-new engineering challenge introduced by the **2026 FIA Technical Regulations**: the mandatory active aerodynamics system, which requires each car to continuously transition between:

- **Z-Mode** — High downforce (used in corners and braking zones)
- **X-Mode** — Low drag (used on high-speed straights)

Unlike the legacy Drag Reduction System (DRS), which only adjusted a single rear wing flap, the 2026 system reconfigures the entire car. The engineering question is no longer *which* mode to use, but **precisely when to switch**. A transition even a fraction of a second too early or too late has a measurable impact on lap time.

### What This Project Does

| Phase | Task | Algorithm | Key Output |
|---|---|---|---|
| 1 | Track Zone Recovery | K-Means Clustering (k=4) | 4 physically-labelled track zones |
| 2 | Driver Anomaly Detection | Isolation Forest (5% contamination) | Anomaly flags on 5% of samples |
| 3 | Aero Mode Prediction (Baseline) | Logistic Regression (L2, balanced) | F1 = 0.9366, AUC = 0.9997 |
| 4 | Aero Mode Prediction (Primary) | Random Forest (200 trees) | F1 = 1.0000, AUC = 1.0000 |

---

## 🏁 Background & Motivation

### The 2026 Active-Aerodynamics Era

The **2026 FIA Technical Regulations (Article 3.10)** mandate moveable aerodynamic elements transitioning between at least two configurations. Key changes include:

- Minimum car mass raised to **798 kg**
- Overall downforce reduced by approximately **30%**
- Re-balanced energy-recovery split
- Full-car aerodynamic reconfiguration replaces the single-flap DRS

The physics is straightforward: above roughly **240 km/h**, the drag penalty of Z-Mode outweighs its cornering benefit. Below this (in corners or braking zones), Z-Mode downforce is essential for grip and stability. The **180–250 km/h transition window** is the critical engineering problem: a tenth of a second of timing improvement per lap compounds across a 53-lap race into multiple seconds of total time gain.

### Why Machine Learning?

In principle, optimal aerodynamic state is deterministic given exact speed, gear, brake state, road gradient, and wheel-slip. In practice, real telemetry is corrupted by:
- GPS noise (±5 km/h on speed readings)
- Wheel-spin artefacts
- Gear-shift transients
- Aerodynamic hysteresis

A learned classifier integrating multiple correlated signals is a far more robust estimator than any single-channel threshold rule.

---

## 📊 Dataset

**Dataset Link:** [F1_2026_JapaneseGP_RedBull.csv](https://github.com/hanan1hub/Predictive-AI-for-Active-AerodynamicsEfficiency-and-Driver-AnomalyAnalysis/blob/main/F1_2026_JapaneseGP_RedBull__2_.csv)

### Properties

| Property | Value |
|---|---|
| Source | 2026 F1 Japanese GP — Red Bull Racing via FastF1 v3.5.x |
| Circuit | Suzuka International Racing Course, Japan |
| Drivers | Max Verstappen (VER) & Isack Hadjar (HAD) |
| Sampling Rate | ≈ 10 Hz per lap |
| Total Samples | 63,673 telemetry records |
| Per-Driver Samples | VER: 32,789 \| HAD: 30,884 |
| Lap Range | Lap 2 → Lap 53 (6 pit/safety laps excluded) |
| Missing Values | 0 — complete dataset |
| Tyre Compounds | MEDIUM (1) and HARD (0) |

Suzuka was chosen for its **figure-of-eight layout** providing a rich mix of zones: flat-out straights (>320 km/h) down to second-gear corners (~90 km/h). Red Bull was selected for data completeness and the contrast between a multi-time world champion (VER) and a rookie (HAD).

### Raw Feature Schema

**Identity & Metadata:** `Driver`, `LapNumber`, `Compound`, `Tire_Age_Laps`, `Time_Elapsed_Sec`, `Distance`

**Vehicle State:** `Speed` (km/h), `RPM`, `nGear` (1–8), `Throttle` (0–100%), `Brake` (boolean), `Acceleration` (m/s²), `Engine_Load`, `Elevation_Delta`

**Spatial:** `X`, `Y`, `Z` (circuit-local coordinates in metres)

**Derived Flags (FastF1):** `Heavy_Braking`, `High_Speed_Zone`, `Gear_Shift_Active`

**Targets:** `Active_Aero_State` (driver's actual decision), `Optimal_Aero` (physics-derived ground truth)

### Target Variables

Two binary targets coexist:

| Target | Description | Z-Mode (0) | X-Mode (1) | Split |
|---|---|---|---|---|
| `Optimal_Aero` | Physics-derived ideal mode | 54,290 | 9,383 | 85.3% / 14.7% |
| `Active_Aero_State` | Driver's actual decision | 35,829 | 27,844 | 56.3% / 43.7% |

The **28-point gap** between X-Mode rates (14.7% vs 43.7%) represents samples where the driver was too early or too late on the transition — the exact phenomenon this project quantifies and predicts.

**`Optimal_Aero` is labelled X-Mode (1) when ALL six conditions hold simultaneously:**
1. `Speed > 240 km/h`
2. `Brake = False`
3. `nGear ≥ 6`
4. `Heavy_Braking = 0`
5. `High_Speed_Zone = 1`
6. `Elevation_Delta > -3`

### Train/Test Split

A **chronological (no-shuffle) 80/20 split** is used to prevent temporal leakage and simulate real deployment:

| Set | Samples | Laps | Z-Mode | X-Mode |
|---|---|---|---|---|
| Train | 50,938 | ~2→43 | 43,578 | 7,360 |
| Test | 12,735 | ~44→53 | 10,712 | 2,023 |

---

## ⚙️ Pipeline Architecture

```
Raw CSV (63,673 × 21)
        │
        ▼
┌─────────────────────────┐
│  Notebook 1             │
│  Data Preprocessing     │  ── Type fixes, IQR outlier detection, winsorisation,
│                         │     physics feature engineering, label construction,
│                         │     StandardScaler, chronological 80/20 split
└────────────┬────────────┘
             │  X_train, X_test, y_train, y_test
             ▼
┌─────────────────────────┐
│  Notebook 2             │
│  Exploratory Data       │  ── Distributions, correlation matrix, driver overlays,
│  Analysis (EDA)         │     circuit map visualisation, bimodality analysis
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Notebook 3             │
│  Modelling              │
│                         │
│  Phase 1: K-Means (k=4) │  ── X, Y, Z, Speed → Track_Zone ∈ {0,1,2,3}
│  Phase 2: Iso. Forest   │  ── 6 driver channels → Anomaly_Flag
│  Phase 3: Log. Reg.     │  ── 18 features → P(X-Mode)  [baseline]
│  Phase 4: Rand. Forest  │  ── 18 features → P(X-Mode)  [primary]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Notebook 4             │
│  Statistical Analysis   │  ── 23 formal hypothesis tests, permutation importance,
│                         │     bootstrap CIs, DeLong/McNemar/Wilcoxon comparisons
└─────────────────────────┘
```

---

## 🔧 Feature Engineering

Four physics-grounded features are derived from raw telemetry using the **FIA-mandated 798 kg car mass**:

```python
# Kinetic Energy — total motion energy of the car (MJ)
Kinetic_Energy_MJ = 0.5 * 798 * (Speed / 3.6)**2 / 1e6

# Longitudinal Force — instantaneous traction or braking force (N)
Longitudinal_Force_N = 798 * (Acceleration / 3.6)

# Energy Efficiency Ratio — high ratio = high-speed coasting = canonical X-Mode condition
Energy_Efficiency_Ratio = Speed / (Engine_Load + 1)

# Rolling Average Speed — 100-sample mean to smooth GPS noise
Speed_Rolling_Avg = Speed.rolling(100).mean()
```

These four engineered features end up accounting for **~32% of the Random Forest's total predictive power**, validating the engineering hypothesis.

### Outlier Handling

| Feature | IQR Outliers | Action |
|---|---|---|
| Speed | 0 | Retained (physically valid) |
| Engine_Load | 0 | Retained |
| Throttle | 0 | Retained |
| RPM | 1,820 (2.86%) | Retained (gear-shift transients — physically meaningful) |
| Acceleration | 11,798 (18.5%) | **Winsorised at ±5σ** (GPS-derivative noise) |
| Elevation_Delta | 8,540 (13.4%) | Retained (physically meaningful extreme events) |

---

## 🤖 Models

### Phase 1 — K-Means Clustering

**Goal:** Recover Suzuka's track zones from GPS + speed, with no human labels.

**Input features:** `X`, `Y`, `Z`, `Speed` (all standardised)

**k selection:** Elbow method across k ∈ {2, …, 9}; inertia bends sharply at **k = 4**

**Results (Silhouette = 0.3754):**

| Zone | Mean Speed | Speed Range | Sample Count | Aero Mode |
|---|---|---|---|---|
| Corner | 139 km/h | 58–223 | 13,025 | Full Z-Mode |
| Braking Zone | 204 km/h | 75–327 | 17,432 | Z-Mode entry (critical) |
| Acceleration Zone | 226 km/h | 129–346 | 12,221 | Transition zone |
| High-Speed Straight | 274 km/h | 204–347 | 20,995 | Primary X-Mode zone |

Each cluster maps to a **physically contiguous portion** of the circuit (verified visually on the X–Y circuit map). The Kruskal-Wallis test confirms the four zones are statistically distinct (H = 34,439, p < 10⁻³⁰⁰, η² = 0.555).

The Track_Zone label produced here is added as the **18th feature** to the supervised models.

---

### Phase 2 — Isolation Forest (Anomaly Detection)

**Goal:** Detect anomalous driver-input patterns without any ground truth label.

**Input features (6 driver-command channels):** `Throttle`, `Brake`, `Acceleration`, `Engine_Load`, `Heavy_Braking`, `Gear_Shift_Active`

**Configuration:** contamination = 0.05, n_estimators = 100, random_state = 42

**Results:** 3,184 anomalies flagged (exactly 5.00%)

| Driver | Anomaly Count | Anomaly Rate |
|---|---|---|
| HAD (Hadjar) | 1,629 | 5.3% |
| VER (Verstappen) | 1,555 | 4.7% |

**Anomaly profile** — anomalous samples are heavy-braking corner-entry events:

| Feature | Anomaly Mean | Normal Mean | Cohen's d |
|---|---|---|---|
| Throttle (%) | 22.7 | 64.6 | -1.01 (Large) |
| Engine_Load | 2,492 | 7,067 | -0.99 (Large) |
| Acceleration (m/s²) | -6.20 | +0.34 | -1.41 (Large) |
| Gear_Shift_Active | 0.711 | 0.022 | +3.96 (Large) |
| Brake | 0.686 | 0.157 | +1.44 (Large) |
| Heavy_Braking | 0.290 | 0.000 | +2.86 (Large) |

---

### Phase 3 — Logistic Regression (Baseline)

**Configuration:** solver=lbfgs, C=1.0 (L2), max_iter=1000, class_weight='balanced', random_state=42

**Top coefficients:**

| Feature | Coefficient | Interpretation |
|---|---|---|
| High_Speed_Zone | +24.92 | FastF1 flag nearly deterministic for X-Mode |
| Brake | -9.30 | Braking immediately disqualifies X-Mode |
| Elevation_Delta | +5.08 | Climbs occur on straight, X-Mode-optimal sections |
| Kinetic_Energy_MJ | -2.29 | Redundant with High_Speed_Zone |
| Speed_Rolling_Avg | +1.69 | Smoothed speed contribution |

**Test Set Performance (n = 12,735):**

| Metric | Score | Project Threshold | Status |
|---|---|---|---|
| Accuracy | 0.9785 | — | ✅ 97.85% |
| Precision | 0.8807 | > 0.90 | ⚠️ Not met (−0.019) |
| Recall | 1.0000 | — | ✅ 100.00% |
| F1-Score | 0.9366 | > 0.90 | ✅ Met (+0.037) |
| AUC-ROC | 0.9997 | — | ✅ 99.97% |

274 false positives, 0 false negatives. LR approximates the AND-rule well when all six conjuncts are correlated — but fails in the ~3% of cases where they diverge (e.g., Speed > 240 but Brake = 1).

---

### Phase 4 — Random Forest (Primary Model)

**Configuration:** n_estimators=200, max_depth=15, min_samples_leaf=5, class_weight='balanced', random_state=42

**Test Set Performance (n = 12,735):**

| Metric | Score | Project Threshold | Status |
|---|---|---|---|
| Accuracy | 1.0000 | — | ✅ 100.00% |
| Precision | 1.0000 | > 0.90 | ✅ Met (+0.100) |
| Recall | 1.0000 | — | ✅ 100.00% |
| F1-Score | 1.0000 | > 0.90 | ✅ Met (+0.100) |
| AUC-ROC | 1.0000 | — | ✅ 100.00% |

Zero false positives, zero false negatives.

**Feature Importance (Mean Decrease in Impurity):**

| Rank | Feature | MDI Importance |
|---|---|---|
| 1 | Speed | 25.4% |
| 2 | Kinetic_Energy_MJ | 19.9% |
| 3 | High_Speed_Zone | 15.7% |
| 4 | Speed_Rolling_Avg | 12.1% |
| 5 | Elevation_Delta | 7.8% |
| 6 | nGear | 6.8% |
| 7 | Throttle | 3.2% |
| 8 | Engine_Load | 2.5% |

Top 8 features account for **93.4%** of total predictive power. Three of the top four are physics-engineered features.

**Permutation importance** (corrected for multicollinearity) identifies the three irreducible drivers: **nGear** (0.167), **Elevation_Delta** (0.162), **Brake** (0.081).

---

## 📈 Results

### Supervised Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9785 | 0.8807 | 1.0000 | 0.9366 | 0.9997 |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Cross-Validation (Stratified 10-Fold)

| Metric | LR Mean ± Std | RF Mean ± Std | Wilcoxon p |
|---|---|---|---|
| Accuracy | 0.9832 ± 0.0022 | 1.0000 ± 0.0001 | 0.0020 |
| F1-Score | 0.9461 ± 0.0068 | 0.9998 ± 0.0002 | 0.0020 |
| AUC-ROC | 0.9994 ± 0.0003 | 1.0000 ± 0.0000 | 0.0020 |

Random Forest wins **all 10 folds** on every metric.

### Bootstrap 95% Confidence Intervals (1,000 resamples)

| Model | Metric | Point Est. | CI Low | CI High |
|---|---|---|---|---|
| LR | Precision | 0.8807 | 0.8680 | 0.8940 |
| LR | F1-Score | 0.9366 | 0.9294 | 0.9433 |
| RF | All metrics | 1.0000 | 1.0000 | 1.0000 |

LR's Precision CI [0.868, 0.894] lies entirely below the 0.90 project threshold — confirming the shortfall is structural, not noise.

### Driver Comparison: VER vs HAD

| Feature | VER Mean | HAD Mean | MW p-value | Cohen's d | Effect |
|---|---|---|---|---|---|
| Speed (km/h) | 219.50 | 216.76 | < 10⁻⁶⁰ | 0.042 | Negligible |
| RPM | 10,648 | 10,588 | < 10⁻⁶⁰ | 0.073 | Negligible |
| Throttle (%) | 63.91 | 61.05 | < 10⁻⁶⁰ | 0.068 | Negligible |
| Aero Deviation Rate | 31.9% | 28.7% | < 10⁻¹⁵ | — | Negligible |

**Key takeaway:** Statistically distinguishable on every feature (due to n ≈ 30,000 per driver), but every Cohen's d < 0.20. The two drivers are **practically indistinguishable** in driving style.

---

## 🧪 Statistical Validation

All **23 formal hypothesis tests** rejected H₀ at α = 0.05 in directions consistent with engineering intuition:

| Test | Purpose | Statistic | p-value | Decision |
|---|---|---|---|---|
| Shapiro-Wilk | Feature normality | W < 0.95 | < 10⁻¹⁰⁰ | Reject (non-normal) |
| D'Agostino-Pearson K² | Feature normality | K² > 5,000 | < 10⁻³⁰⁰ | Reject |
| Mann-Whitney U (driver) | Same speed distribution | U → 0 | < 10⁻⁶⁰ | Reject |
| t-test (driver) | Equal means | t > 5 | < 10⁻⁶ | Reject |
| Levene's test | Equal variances | F > 10 | < 10⁻³⁰ | Reject |
| χ² (driver × deviation) | Independent | χ² = 73.1 | < 10⁻¹⁵ | Reject (V = 0.034) |
| Pearson r (target) | r = 0 | r ≥ 0.25 | < 10⁻³⁰⁰ | Reject (6 features) |
| Spearman ρ (target) | ρ = 0 | ρ ≥ 0.27 | < 10⁻³⁰⁰ | Reject (6 features) |
| Kruskal-Wallis (zones) | Same speed in 4 zones | H = 34,439 | < 10⁻³⁰⁰ | Reject (η² = 0.555) |
| One-way ANOVA (zones) | Same speed in 4 zones | F = 26,474 | < 10⁻³⁰⁰ | Reject |
| Dunn post-hoc | Zone pair i = j | Bonferroni-adj. | < 10⁻³⁰⁰ | Reject all 6 pairs |
| χ² (anomaly × deviation) | Independent | χ² = 817.4 | ≈ 10⁻¹⁸⁰ | Reject (V = 0.113) |
| Fisher's exact | Independent | OR = 0.18 | ≈ 10⁻²²⁶ | Reject |
| **DeLong AUC** | AUC(LR) = AUC(RF) | z = -2.062 | **0.0392** | Reject |
| **McNemar** | Same error rate | χ² = 272.0 | < 10⁻⁶⁰ | Reject |
| **Wilcoxon signed-rank (CV)** | Same per-fold F1 | W = 0 | **0.0020** | Reject |
| Permutation imp. (nGear) | Importance = 0 | t = 153.3 | < 10⁻¹⁰⁰ | Reject |
| Permutation imp. (Elevation) | Importance = 0 | t = 234.9 | < 10⁻¹⁰⁰ | Reject |
| Permutation imp. (Brake) | Importance = 0 | t = 128.0 | < 10⁻¹⁰⁰ | Reject |

**McNemar analysis:** Every single sample on which LR and RF disagreed (n = 274) was one where RF was correct and LR was wrong. RF strictly dominates LR — not just in aggregate.

---

## 💡 Key Findings

### 1. Why Random Forest Beats Logistic Regression

`Optimal_Aero` is a **multiplicative conjunction** (Speed > 240 AND Brake = 0 AND nGear ≥ 6 AND …). Logistic Regression can only approximate AND-rules through weighted sums, which works in the 97% of cases where all conjuncts are correlated. The 274 LR false positives are exactly the 3% of cases where they diverge (e.g., Speed > 240 but Brake = 1). Random Forest partitions axis-aligned on each conjunct, recovering the exact rule.

### 2. The Anomaly–Deviation Sign Reversal

The odds ratio is 0.18 (anomalies are *less* likely to coincide with timing deviations). Why? Timing deviations occur in the 180–260 km/h transition window — these are routine, mid-throttle samples that look *normal* to Isolation Forest. Anomalies concentrate in heavy-braking corner-entry events, deep in the Z-Mode regime where both driver and physics agree on Z-Mode. The anomaly detector catches **unusual driver aggression**, not **mis-timed aero switching**. Both are useful signals, but they are not the same signal.

### 3. Statistically Significant vs Practically Meaningful

VER vs HAD comparison illustrates a classical big-data pitfall: with n ≈ 30,000 per driver, even a 2.74 km/h speed difference (219.5 vs 216.8 km/h) is highly significant (p < 10⁻⁶⁰), but Cohen's d = 0.042 is far below any practical threshold. The two drivers are **statistically distinguishable but practically identical** in driving style.

### 4. Physics Engineering Validates Itself

Three of the top four MDI features are engineered (`Kinetic_Energy_MJ`, `Speed_Rolling_Avg`, `High_Speed_Zone`-related). Their combined importance of ~47% confirms that physics-grounded features dominate raw sensor readings.

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Single circuit:** All results are Suzuka-specific. Generalisation to Monza (high-speed, minimal corners) or Monaco (near-zero straights) requires fresh training.
- **Approximate breakeven threshold:** The 240 km/h X-Mode threshold is an engineering approximation, not a chassis-specific wind-tunnel value. A ±10 km/h shift changes the label distribution significantly.
- **Fixed anomaly contamination:** contamination = 0.05 is set by domain assumption, not cross-validation. Tunable contamination would improve the downstream association analysis.

### Planned Extensions

| Extension | Description |
|---|---|
| **Multi-circuit generalisation** | Re-train on Monza, Monaco, Spa; test zone-structure transferability |
| **Temporal modelling** | LSTM/Transformer over a 50-sample rolling window for forward-looking prediction (predict N samples ahead) |
| **Tyre-state conditioning** | Stratify by compound and tyre age to surface compound-specific aero thresholds |
| **Real-time edge deployment** | Compile 14 MB Random Forest to TensorFlow-Lite / ONNX for on-car micro-controller, targeting < 5 ms latency |
| **Anomaly target reformulation** | Define anomaly target around transition-window deviations to recover the expected 3.5× association strength |

### Optional Deep Learning Sanity Check

A two-layer MLP (hidden sizes 64/32, ReLU, dropout 0.2, batch norm, Adam, 50 epochs) scored **F1 = 0.9854 / AUC = 0.9991** — sitting between LR and RF, but without interpretability. RF remains the preferred production model for tabular telemetry data.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── F1_2026_JapaneseGP_RedBull.csv      # Raw compiled telemetry (63,673 × 21)
│   └── df_compiled.csv                      # Preprocessed output of Notebook 1
│
├── notebooks/
│   ├── 01_preprocessing.ipynb              # Data loading, cleaning, feature engineering, split
│   ├── 02_eda.ipynb                         # Distributions, correlation matrix, circuit maps
│   ├── 03_modelling.ipynb                   # K-Means, Isolation Forest, LR, Random Forest
│   └── 04_statistical_analysis.ipynb        # All 23 hypothesis tests + dashboard
│
├── models/
│   ├── kmeans_k4.pkl                        # Trained K-Means model
│   ├── isolation_forest.pkl                 # Trained Isolation Forest
│   ├── logistic_regression.pkl              # Trained Logistic Regression
│   └── random_forest_200.pkl                # Trained Random Forest
│
├── artefacts/
│   ├── standard_scaler.pkl                  # Fitted StandardScaler (train only)
│   ├── X_train.npy / X_test.npy             # Feature splits
│   ├── y_train.npy / y_test.npy             # Label splits
│   └── active_aero_state.npy               # Driver-actual target array
│
├── stats_outputs/                           # CSV exports of all 23 test results
│
├── figures/                                 # All report figures (Fig. 1–24)
│
└── README.md
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install fastf1==3.5.0 scikit-learn pandas numpy matplotlib seaborn scipy

# Run the full pipeline in order
jupyter notebook notebooks/01_preprocessing.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_modelling.ipynb
jupyter notebook notebooks/04_statistical_analysis.ipynb
```

**Python version:** 3.9+  
**Key dependencies:** `fastf1`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

---

## 📚 References

1. FIA, "2026 Formula One Technical Regulations, Art. 3.10," October 2024.
2. Heilmeier et al., "Minimum curvature trajectory planning and control for an autonomous race car," *Vehicle System Dynamics*, 2020.
3. Stoll et al., "Tyre wear prediction for formula racing using machine learning," *ITSC*, 2021.
4. Valls et al., "Fuel strategy optimization using machine learning in Formula 1," *ICPRAM*, 2021.
5. Casas & Vicén, "LSTM-based lap time prediction for Formula 1," *Journal of Sports Engineering and Technology*, 2022.
6. Balaji et al., "Overtaking probability prediction in F1 using ensemble learning," *IEEE Big Data*, 2020.
7. Liu, Ting & Zhou, "Isolation forest," *IEEE ICDM*, 2008.
8. Zimek et al., "A survey on unsupervised outlier detection in high-dimensional numerical data," *Statistical Analysis and Data Mining*, 2012.
9. Paefgen et al., "Evaluation and aggregation of pay-as-you-drive insurance rate factors," *IEEE VTC*, 2012.
10. Oehrly, "FastF1: A Python package for accessing F1 data," GitHub, 2024.
11. DeLong et al., "Comparing the areas under two or more correlated receiver operating characteristic curves," *Biometrics*, 1988.
12. Breiman, "Random forests," *Machine Learning*, 2001.
13. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, 3rd ed., O'Reilly, 2022.
14. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 2011.

---

*Aero Intelligence — CS-245 Machine Learning Final Project, NUST 2026*
