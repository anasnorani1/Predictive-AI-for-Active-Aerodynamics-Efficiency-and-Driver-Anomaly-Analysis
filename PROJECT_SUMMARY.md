# CS-245 Machine Learning Project — Complete Summary
## Predictive AI for Active Aerodynamics Efficiency and Driver Anomaly Analysis

**Team:** Aero Intelligence — Hanan Majeed | Maha Mohsin | Anas Norani  
**Course:** CS-245 — Machine Learning | Instructor: Mam Nazia Pervaiz  
**Dataset:** 2026 F1 Japanese Grand Prix — Red Bull Telemetry (Suzuka)  
**Kaggle:** https://www.kaggle.com/datasets/mehe6242/f1-2026-active-aero-telemetry-red-bull  

---

## What the Project Does

The 2026 F1 season replaced the old DRS with Active Aerodynamics — front and rear wings that switch between Z-Mode (high downforce for corners) and X-Mode (low drag for straights). This project builds an AI system that answers two questions:
1. **When** is the physics-optimal moment to switch aero modes?
2. **Why** did the driver deviate from that optimal moment?

---

## Dataset

| Property | Value |
|---|---|
| Total rows | 63,673 |
| Columns (raw) | 21 |
| Engineered features | 4 (Kinetic Energy, Longitudinal Force, EER, Speed Rolling Avg) |
| Drivers | VER (Max Verstappen), HAD (Isack Hadjar) |
| Laps | 52 (Laps 2–53) |
| Target | Active_Aero_State (0=Z-Mode, 1=X-Mode) |
| Training target | Optimal_Aero (physics-derived, NOT driver-actual) |
| Missing values | Zero |
| Class balance | ~56% Z / 44% X — no resampling needed |
| Max speed | 347 km/h |

**Raw columns:** Driver, LapNumber, Compound, Speed, RPM, nGear, Throttle, Brake, X, Y, Z, Distance, Tire_Age_Laps, Time_Elapsed_Sec, Acceleration, Engine_Load, Elevation_Delta, Heavy_Braking, Active_Aero_State, High_Speed_Zone, Gear_Shift_Active

---

## Key Design Principle

The supervised models are trained on a **physics-derived label** (`Optimal_Aero`), NOT on `Active_Aero_State` (what the driver actually did). After training, model predictions are compared against the driver's actual column — the gap is the detected timing error.

**X-Mode (1) is optimal when ALL of:**
- Speed > 240 km/h
- Brake = False
- nGear ≥ 6
- Heavy_Braking = 0
- High_Speed_Zone = 1
- Elevation_Delta > −3

---

## Four-Phase ML Pipeline

| Phase | Model | Type | Input | Output | Metric |
|---|---|---|---|---|---|
| 1 | K-Means (k=4) | Unsupervised | X, Y, Z, Speed | Track_Zone label | Silhouette Score (target > 0.60) |
| 2 | Isolation Forest | Unsupervised | Throttle, Brake, Acceleration, Engine_Load, Heavy_Braking, Gear_Shift_Active | Anomaly flag + score | Temporal correlation with timing errors |
| 3 | Logistic Regression (L2) | Supervised | 18 features (17 + Track_Zone) | Optimal_Aero prediction + coefficients | Accuracy, F1, AUC-ROC (target F1 > 0.90) |
| 4 | Random Forest (n=200, depth=15) | Supervised | 18 features (17 + Track_Zone) | Optimal_Aero prediction + importance | Accuracy, F1, AUC-ROC (target F1 > 0.90) |

---

## Preprocessing Steps (Notebook 1)

1. Load CSV → copy raw dataframe
2. Missing value check (dataset is complete — no imputation)
3. Brake column: string `'True'/'False'` → integer 0/1
4. Compound: `'MEDIUM'/'HARD'` → label-encoded 0/1 (`Compound_Encoded`)
5. Binary columns cast to int: `Active_Aero_State`, `High_Speed_Zone`, `Heavy_Braking`, `Gear_Shift_Active`
6. Outlier detection via IQR; Acceleration winsorised at ±5σ; all others retained (physically valid)
7. Feature engineering: `Kinetic_Energy_MJ`, `Longitudinal_Force_N`, `Energy_Efficiency_Ratio`, `Speed_Rolling_Avg`
8. Construct `Optimal_Aero` label (physics rule above)
9. `StandardScaler` on all 17 numeric features
10. **Chronological 80/20 train/test split** (no shuffle — prevents temporal leakage)
11. Save artefacts to `artefacts/` folder: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `standard_scaler.pkl`, `feature_cols.pkl`, `df_preprocessed.csv`, `df_train.csv`, `df_test.csv`

---

## EDA Plots Generated (Notebook 2)

- Feature distributions (histogram grid — all 17+ features)
- Driver comparison (VER vs HAD): speed, throttle, kinetic energy
- Target variable pie charts (Optimal_Aero vs Active_Aero_State)
- Lap-by-lap X-Mode activation rate
- Speed distribution by aero mode (with 240 km/h threshold line)
- Speed vs RPM scatter coloured by aero state
- Gear distribution by aero mode
- Throttle & brake usage patterns
- Pearson correlation heatmap (full matrix)
- Feature correlation bar chart with Optimal_Aero
- Suzuka 2D track map coloured by speed + by aero state
- Elevation profile (one full lap)
- Engineered features vs speed scatter plots
- Engineered features boxplots by aero mode
- Tire degradation vs speed per driver
- Compound comparison bar chart
- Single-lap telemetry deep dive with deviation timeline
- Pairplot (top 5 features, 5,000-sample)
- Mann-Whitney U significance tests for all key features

---

## Modelling Details (Notebook 3)

**K-Means:**
- Input: X, Y, Z, Speed (StandardScaled independently)
- k=4 selected via Elbow Method
- Cluster labels auto-assigned by mean speed ranking
- Track_Zone appended to full dataframe and then to feature set (making 18 total features)
- Final feature re-scaled with a new StandardScaler fitted on train only

**Isolation Forest:**
- Input: Throttle, Brake, Acceleration, Engine_Load, Heavy_Braking, Gear_Shift_Active
- contamination=0.05 (5% expected anomaly rate)
- Outputs: `Anomaly_Flag` (−1/+1), `Anomaly_Score`
- Anomalies correlated with `Aero_Deviation` column
- Root cause = feature with largest mean delta between anomaly vs normal groups

**Logistic Regression:**
- 18 features, L2 regularisation, C=1.0, class_weight='balanced', max_iter=1000
- Outputs: class predictions + probability scores
- Interpretability via feature coefficients

**Random Forest:**
- 18 features, n_estimators=200, max_depth=15, class_weight='balanced', min_samples_leaf=5
- Outputs: class predictions + probability scores
- Feature importance via Mean Decrease in Impurity (MDI)

**Evaluation for supervised models:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix, Classification Report, ROC Curve

---

## Saved Artefacts

```
artefacts/
  df_preprocessed.csv       ← Full preprocessed dataframe
  df_train.csv              ← Training split (80%)
  df_test.csv               ← Test split (20%)
  X_train.npy               ← Scaled training features
  X_test.npy                ← Scaled test features
  y_train.npy               ← Training labels (Optimal_Aero)
  y_test.npy                ← Test labels (Optimal_Aero)
  y_actual_full.npy         ← Driver's actual aero state (full dataset)
  standard_scaler.pkl       ← Fitted StandardScaler
  feature_cols.pkl          ← Feature column name list
  df_final_with_anomalies.csv ← Final dataframe with anomaly flags
  model_comparison_table.csv  ← All model metrics in one CSV

models/
  kmeans_k4.pkl             ← K-Means model
  isolation_forest.pkl      ← Isolation Forest model
  logistic_regression.pkl   ← Logistic Regression model
  random_forest.pkl         ← Random Forest model
```

---

## CS-245 Requirements Checklist

| Requirement | Status | How Met |
|---|---|---|
| ≥ 2 supervised models | ✅ FULFILLED | Logistic Regression + Random Forest |
| ≥ 1 unsupervised model | ✅ EXCEEDED | K-Means + Isolation Forest |
| Comparative analysis | ✅ FULFILLED | Full metrics table + ROC curves + bar chart |
| No deep learning | ✅ COMPLIANT | No LSTM or neural networks |
| Bonus Streamlit app | PLANNED | To be built after notebooks |
| Classification metrics | ✅ FULFILLED | Accuracy, Precision, Recall, F1, AUC-ROC |
| Clustering metrics | ✅ FULFILLED | Silhouette Score + visual track map |
| Visualisations | ✅ FULFILLED | 20+ plots across all notebooks |
| Data preprocessing | ✅ FULFILLED | Full pipeline in Notebook 1 |
| EDA | ✅ FULFILLED | 20+ analysis plots in Notebook 2 |

---

## How to Run

1. Place `F1_2026_JapaneseGP_RedBull__2_.csv` in the same directory as the notebooks
2. Run `01_Data_Preprocessing.ipynb` (generates `artefacts/` folder)
3. Run `02_EDA.ipynb` (reads from `artefacts/`)
4. Run `03_Modelling.ipynb` (reads from `artefacts/`, saves to `models/`)

**Dependencies:** numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, joblib
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
```
