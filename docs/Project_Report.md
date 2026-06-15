# Project Report
## Anticipatory Aerodynamic Mode Prediction in Formula 1 Using Machine Learning

**Submitted by:** Anas Norani, Hanan Majeed, Maha Mohsin
**Program:** BS Data Science, SEECS, NUST
**Date:** June 2026

---

## 1. What This Project Is About (Plain Language Summary)

Formula 1 cars in 2026 have a new feature called **Active Aerodynamics**. The car can switch between two modes:
- **Z-Mode (High Downforce):** Wings push the car into the road for better grip in corners.
- **X-Mode (Low Drag):** Wings flatten out on straights so the car goes faster.

The switch happens automatically when the car reaches a certain speed (240 km/h) in a high gear (gear 6 or above). The problem is that this switch is not instant — it takes a fraction of a second for the system to react. By the time the car detects it *should* switch, it may have already missed the optimal moment.

**Our solution:** Instead of reacting when the threshold is reached, can a computer *predict* 1 second ahead that the car is *about to reach* X-Mode conditions? If yes, the system can start switching early.

We treated this as a machine learning problem: given the last 5 seconds of car data (speed, gear, throttle, position, etc.), predict what mode the car will need **1 second from now**.

---

## 2. Why This is Harder Than It Sounds

A naïve approach — "predict the current mode from current data" — is completely useless. The mode is defined by speed and gear, which are both in the input data. The model would just learn a simple rule and report 100% accuracy. That tells us nothing useful.

We solved this by **shifting the target into the future**: the model sees data from seconds *t−5s* to *t*, but must predict what mode is needed at time *t+1s*. The future data is **completely hidden** from the model. This forces it to genuinely learn the car's approach patterns rather than memorise a threshold rule.

---

## 3. Data Used

We downloaded real 2025/2026 Formula 1 telemetry data using the open-source **FastF1** Python library. The data includes measurements at approximately 10 times per second for every lap.

| Circuit | Race | Driver(s) | Rows | X-Mode Rate |
|---|---|---|---|---|
| **Monza** | 2025 Italian GP | Verstappen | 31,098 | 57% |
| **Monaco** | 2025 Monaco GP | Verstappen | 39,881 | 13% |
| **Silverstone** | 2025 British GP | Verstappen | 25,060 | 38% |
| **Suzuka** | 2026 Japanese GP | Verstappen + Hadjar | 63,673 | 35% |

**Total:** 159,712 data rows → approximately 145,000 training/test windows at the primary 1-second horizon.

**Features used (12 total):**
- Raw measurements: Speed, RPM, Gear, Throttle, Brake, X/Y/Z position
- Computed physics: Acceleration, Elevation change, Kinetic Energy, Longitudinal Force

We intentionally excluded any data about the *future* state of the car. Using future speed or gear to predict future mode would be cheating.

---

## 4. How We Tested Fairly (Leave-One-Circuit-Out)

Instead of splitting one dataset randomly into train/test, we used **Leave-One-Circuit-Out (LOCO)** evaluation:

- Train on 3 circuits → Test on the 4th circuit
- Repeat 4 times, each circuit gets a turn as the test set
- This tests whether a model learned *general* racing physics, or just memorised one specific track

This is a much harder and more realistic test than a random split.

---

## 5. Models Tested

We tested 8 models from simple to complex:

**Classical (no temporal reasoning):**
- **LR-instant** — Logistic Regression using only the single most recent data point
- **RF-instant** — Random Forest using only the single most recent data point

**Classical (with time history):**
- **RF-lag** — Random Forest given the full 5-second window of 50 time steps (flattened into 600 input values)

**Deep Learning (temporal):**
- **CNN** — 1D Convolutional Neural Network (~136k parameters)
- **LSTM** — Long Short-Term Memory recurrent network (~213k parameters)
- **GRU** — Gated Recurrent Unit (~162k parameters)
- **TCN** — Temporal Convolutional Network with dilated convolutions (~152k parameters)
- **Transformer** — Self-attention encoder with causal masking (~68k parameters)

All deep models were trained on an NVIDIA P100 GPU via Kaggle, using focal loss (to handle class imbalance), early stopping, and automatic mixed precision.

---

## 6. Results

### 6.1 Primary Results: 1-Second Ahead Prediction (H=10)

The table below shows mean performance across all 4 circuits. Higher AUC-ROC is better (1.0 = perfect, 0.5 = random guessing).

| Model | F1-Score | AUC-ROC | AUC-PR | Circuit Stability (σ) |
|---|---|---|---|---|
| **LR-instant** | **0.906** | **0.963** | **0.881** | 0.021 (most stable) |
| RF-instant | 0.872 | 0.959 | 0.864 | 0.011 |
| RF-lag | 0.794 | 0.945 | 0.815 | 0.026 |
| **Transformer** | 0.779 | **0.911** | 0.783 | 0.055 *(best deep model)* |
| LSTM | 0.756 | 0.878 | 0.729 | 0.085 |
| GRU | 0.768 | 0.877 | 0.724 | 0.104 |
| TCN | 0.785 | 0.835 | 0.755 | 0.201 |
| CNN | 0.748 | 0.818 | 0.712 | 0.111 |

**Key finding:** The simplest model (Logistic Regression) is the best overall. Deep learning does not help here — and in some cases makes things significantly worse.

### 6.2 Circuit-by-Circuit Breakdown

| Model | Monaco | Monza | Silverstone | Suzuka |
|---|---|---|---|---|
| LR-instant | **0.980** | 0.932 | 0.970 | 0.969 |
| RF-instant | 0.955 | **0.963** | **0.972** | 0.945 |
| RF-lag | 0.907 | 0.947 | 0.963 | **0.961** |
| **Transformer** | **0.956** | 0.871 | 0.961 | 0.857 |
| LSTM | 0.927 | 0.863 | 0.957 | 0.765 |
| **GRU** | **0.728** ← collapse | 0.896 | **0.968** | 0.917 |
| TCN | 0.944 | 0.890 | **0.968** | 0.537 ← collapse |
| CNN | 0.811 | 0.821 | 0.956 | 0.684 |

The two most important patterns are highlighted:
- **GRU collapses at Monaco** (0.728 AUC-ROC, F1-xmode = 0.005) — basically never predicts X-Mode
- **TCN collapses at Suzuka** (0.537 AUC-ROC — near-random)
- **Transformer stays robust at Monaco** (0.956) while GRU fails catastrophically

### 6.3 How Hard is Predicting Further Ahead?

| Model | H=1 (0.1s) | H=5 (0.5s) | H=10 (1s) | H=25 (2.5s) | H=50 (5s) |
|---|---|---|---|---|---|
| LR-instant | 0.999 | 0.993 | 0.963 | 0.786 | 0.471 |
| RF-instant | 0.999 | 0.993 | 0.959 | 0.777 | 0.562 |
| RF-lag | 0.999 | 0.985 | 0.945 | 0.745 | 0.466 |
| Transformer | 0.999 | 0.987 | 0.911 | 0.691 | **0.595** |
| GRU | 0.999 | 0.986 | 0.877 | 0.682 | 0.505 |

**Interpretation:**
- At 0.1s and 0.5s: all models are near-perfect (the car hasn't changed much in 50ms)
- At 1s: meaningful differences appear — this is the useful operating range
- At 2.5s+: all models degrade badly
- At 5s: basically random guessing for all models

**Practical implication:** Real-time aero systems should operate with a ~1 second look-ahead window. Beyond 2.5 seconds, no current method is reliable enough.

---

## 7. Why the Simple Model Wins: The Circuit Memorisation Problem

This is the most important finding of the project, and it is counter-intuitive.

**Why does Logistic Regression beat a GRU with 162,000 parameters?**

The answer is **circuit memorisation**:

- **RF-lag** has access to the full 5-second speed history. It learns: *"When speed has been around 240-260 km/h for the last 5 seconds and is climbing, we're probably approaching a fast straight."* This rule works perfectly on Monza and Silverstone. But on **Monaco**, the entire circuit is slower — top speed is around 200-220 km/h. The "fast straight" signal never fires, so the model barely predicts X-Mode at all.

- **GRU** learns an even more compressed version of the same mistake. Its internal memory (hidden state) stores a summary of the speed trajectory. After training on fast circuits, it has learned: *"if the speed trajectory was climbing steeply, X-Mode is coming."* At Monaco, that pattern never appears. The result: GRU predicts F1-xmode = **0.005** at Monaco — essentially stuck in Z-Mode prediction permanently.

- **Logistic Regression** has no memory. It only looks at the *current* instant. *"Is speed ≥ 240 km/h right now AND gear ≥ 6?"* The approach to that threshold looks the same regardless of which circuit you're on. Hence LR remains robust.

This is confirmed by our interpretability analysis (Section 8).

---

## 8. What the Models Actually Look At (Interpretability)

We used **Integrated Gradients** — a method that traces which input features most influenced each prediction — on the Monaco fold (the hardest test).

| Feature | GRU importance | Transformer importance |
|---|---|---|
| Longitudinal Force | **#1** (9.9%) | #5 (3.7%) |
| Acceleration | **#2** (9.7%) | #6 (3.3%) |
| Throttle | #3 (8.5%) | **#1** (4.8%) |
| Speed | #4 (8.1%) | **#2** (4.1%) |
| nGear | #8 | **#3** (3.9%) |
| RPM | #9 | **#4** (3.9%) |

- **GRU** relies on dynamic (rate-of-change) features: Force and Acceleration tell it *how fast the car is changing state*. These are calibrated to high-speed dynamics at training circuits and fail at Monaco's slower pace.

- **Transformer** relies on instantaneous state features: Throttle, Speed, nGear, RPM tell it *what state the car is in right now*. These signals transfer across circuits because the threshold for X-Mode (speed 240 km/h + gear 6) looks the same regardless of what the overall circuit speed envelope is.

We also produced **attention maps** for the Transformer. They show that the model focuses mainly on the most recent 5–10 time steps, with a peak at the most recent sample. This pattern is consistent across Monaco and high-speed circuits, confirming that the Transformer does not develop circuit-specific attention patterns.

---

## 9. Transition-Window Analysis

Mode changes are rare events. We specifically analysed the ±0.5 seconds around each actual mode change (transition windows) versus windows where the mode is stable.

| Circuit | Model | F1 at Transitions | F1 at Stable | Drop |
|---|---|---|---|---|
| Monaco | Transformer | 0.496 | 0.902 | −0.406 |
| Monaco | GRU | 0.328 | 0.480 | −0.152 |
| Monza | Transformer | 0.551 | 0.865 | −0.314 |
| Monza | GRU | 0.615 | 0.943 | −0.327 |
| Silverstone | Transformer | 0.659 | 0.970 | −0.311 |
| Silverstone | GRU | 0.674 | 0.973 | −0.299 |
| Suzuka | Transformer | 0.554 | 0.599 | −0.045 |
| Suzuka | GRU | 0.575 | 0.860 | −0.285 |
| **Mean** | **Transformer** | **0.565** | **0.834** | **−0.269** |
| **Mean** | **GRU** | **0.548** | **0.814** | **−0.266** |

Even the best deep model drops ~0.41 F1 at the exact moment of transition. This is expected — the model is predicting a future state that the current telemetry has not yet fully "announced." It suggests that future work should use loss functions that penalise transition-window errors more heavily.

---

## 10. Code and Pipeline

The full pipeline consists of 5 scripts:

| Script | Purpose |
|---|---|
| `01_build_anticipatory_dataset.py` | Downloads F1 telemetry via FastF1, engineers features, builds sliding windows |
| `02_baselines.py` | Trains and evaluates LR-instant, RF-instant, RF-lag for all circuits and horizons |
| `03_deep_models.py` | Trains CNN, LSTM, GRU, TCN, Transformer with GPU support, checkpointing, early stopping |
| `04_evaluate.py` | Generates horizon sweep tables, bootstrap confidence intervals, LaTeX output |
| `05_xai.py` | Integrated Gradients heatmaps, Transformer attention maps, SHAP for RF-lag |

Training was run on Kaggle's free GPU tier (NVIDIA P100, 16 GB VRAM). Total GPU time ≈ 6–8 hours for all 52 deep model folds (5 models × 4 circuits + 2 models × 4 circuits × 4 additional horizons for sweep).

All results are reproducible with `seed = 42`.

---

## 11. What We Would Do Next

1. **Circuit-adaptive normalisation:** Replace absolute Speed with "speed relative to circuit average." This would remove the calibration anchor that causes GRU and RF-lag to fail at Monaco.

2. **Domain-adversarial training:** Add a circuit-classifier head that is trained adversarially, forcing the main model to learn circuit-independent features.

3. **Transition-aware loss:** Weight the loss function more heavily near mode-transition windows to improve the single most difficult prediction scenario.

4. **More circuits and drivers:** Four circuits is a small test bed. Extending to 10+ circuits from a full season would make the LOCO protocol much stronger.

5. **Edge deployment:** The models are small enough (68k–213k parameters) to run on embedded hardware. Measuring actual latency on an F1-grade ECU would confirm real-time feasibility.

6. **Outcome-based label:** Our X-Mode label is based on a physics rule (speed threshold + gear threshold). A better label would come from a lap-time or energy simulation: *"would switching to X-Mode at this moment have improved lap time?"*

---

## 12. Summary

| Question | Answer |
|---|---|
| Can we predict aero mode 1 second ahead? | Yes — AUC-ROC 0.963 (LR) to 0.818 (CNN) |
| Does deep learning help? | No — simpler models are more robust |
| Why does complex temporal modelling fail? | Circuit memorisation: models learn circuit-specific speed patterns, not general physics |
| Which deep model is best? | Transformer (most circuit-robust, AUC-ROC 0.911) |
| What is the useful prediction horizon? | ≤1 second; beyond 2.5s all methods fail |
| What features matter most? | Throttle, Speed, nGear (instantaneous state) — not dynamics |
| What is the hardest scenario? | Monaco street circuit + transition windows |

---

## References

- FIA 2026 Formula 1 Technical Regulations, Art. 3.10
- Oehrly, T. FastF1: A Python package for F1 data (2024)
- Lin et al. "Focal loss for dense object detection." ICCV (2017)
- Bai et al. "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling." arXiv:1803.01271 (2018)
- Vaswani et al. "Attention is all you need." NeurIPS (2017)
- Sundararajan et al. "Axiomatic attribution for deep networks (Integrated Gradients)." ICML (2017)
