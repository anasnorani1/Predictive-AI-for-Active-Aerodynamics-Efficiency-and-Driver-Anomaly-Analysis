# Aero Intelligence — Redesign & Execution Plan
## A 3-week sprint to a submittable IEEE conference paper (deadline: 21 June)

*This plan is built around what is actually finishable in 21 days. It reuses everything you already have — FastF1 pipeline, features, statistics code, and ~80% of your written paper — and fixes the one fatal flaw (the circular label) by changing **what** you predict, not throwing the project away.*

---

## 1. The core idea (read this first — everything depends on it)

### 1.1 What was broken
Your old target was `Optimal_Aero[t] = 1 iff Speed[t] ≥ 240 AND nGear[t] ≥ 6`, predicted from features that **include** `Speed[t]` and `nGear[t]`. The model was handed the answer. F1 = 0.98 meant nothing.

### 1.2 The fix: predict the **future**, not the present
New task — **anticipatory aerodynamic mode prediction**:

> Given a sliding window of the **past** W telemetry samples `X[t−W+1 … t]` (which does **not** contain the target), predict the physics-optimal aero mode **H steps in the future**: `y = Optimal_Aero[t + H]`.

Because the future speed/gear are **not** in the input window, the model can no longer "look up" the answer. It must **forecast the regime** — i.e., learn that a particular braking-then-throttle-then-upshift pattern means "you'll be on the straight and need X-Mode in 1.5 seconds." That is a real, non-trivial sequential prediction problem.

### 1.3 Why this single change fixes the paper

| Old problem | How the reframe fixes it |
|---|---|
| Circular/leaky label | Future target not in input window → genuine forecasting |
| No reason to use DL | Anticipation needs temporal memory → LSTM/TCN/Transformer have a principled advantage |
| "RF beats LR" was a geometric artifact | Now sequence-DL beats RF for a *real* reason (RF has no time axis) |
| Stats applied to a tautology | Same tests, now applied to a real result |
| Single circuit, no generalization | Add 2–3 circuits and do **cross-circuit** evaluation |
| No practical value | Real actuators have latency; *anticipation* (acting before the threshold) is exactly what a control system needs |

### 1.4 Your new one-sentence contribution
> *"We reframe active-aero mode selection as **anticipatory time-series classification** — forecasting the physics-optimal configuration H steps ahead from a telemetry window — and show that temporal deep models (TCN/Transformer) substantially outperform instantaneous and lag-augmented classical baselines, with the advantage growing as the prediction horizon and circuit-shift increase."*

This is honest, defensible, DL-centred, and finishable in 3 weeks.

---

## 2. Data plan (Days 1–2)

You already pull FastF1. Two cheap, high-value additions:

1. **Multi-circuit data.** Pull 3–4 circuits with *different speed profiles* so cross-circuit generalization is testable and meaningful:
   - **Monza** (low-downforce, long straights — X-Mode dominant)
   - **Monaco** (street, slow — Z-Mode dominant)
   - **Silverstone / Spa** (mixed, high-speed corners)
   - **Suzuka** (your existing data — keep it)
   Pull both Red Bull drivers per race (or more drivers — more diversity is strictly better and free).
2. **Keep your preprocessing** (type fixes, IQR retention of real extremes, Winsorising acceleration). It's fine.

**Splitting strategy (this is what makes the result credible):**
- **Within-circuit split** (chronological, like before) — for the "easy" comparison.
- **Leave-one-circuit-out (LOCO)** — train on 3 circuits, test on the held-out 4th. This is your *generalization headline*. Rotate so every circuit is held out once.
- **No leakage:** never let the same lap appear in train and test; windows must not straddle the split boundary.

**Working isolation:** put all new multi-circuit notebooks, configs, and derived artefacts under `multi_circuit_work/` so the existing Suzuka-only `artefacts/`, `graphs/`, and `models/` directories stay untouched.

---

## 3. Task formulation & windowing (Day 3)

### 3.1 Sliding-window construction
For each timestep `t`, build:
- **Input** `X_t ∈ ℝ^(W × F)`: the last `W` samples of the `F` telemetry channels (raw + your physics features KE, Force, EER), **excluding** the two target-defining columns *at the prediction target time* (keep current speed/gear in the window — they're legitimate past observations; just not the future ones).
- **Label** `y_t = Optimal_Aero[t + H]`.

Pseudocode:
```python
W, H = 20, 15      # 2.0 s window, 1.5 s horizon at 10 Hz (tune these)
X, y = [], []
for t in range(W-1, len(df) - H):
    if same_lap(t-W+1, t+H):        # no straddling laps/sessions
        X.append(df[feature_cols].iloc[t-W+1 : t+1].values)   # (W, F)
        y.append(int(df['Optimal_Aero'].iloc[t + H]))
X = np.stack(X); y = np.array(y)     # X: (N, W, F)
```

### 3.2 The horizon sweep (your difficulty knob — and a great analysis)
Run the whole study at multiple horizons, e.g. `H ∈ {1, 5, 10, 15, 20, 30}` (0.1 s → 3 s ahead).
- At small H the task is near-trivial (everyone scores high) → confirms your pipeline works.
- At large H it becomes genuinely hard → this is where DL pulls away from classical baselines.
- **The "accuracy vs horizon" curve is one of your best figures**: it visually proves the task is non-circular (performance *degrades* with horizon) and that DL degrades *slower*.

---

## 4. The algorithms — every model explained

You will run a **ladder of models** from simplest to strongest. The story is the *gap* between them.

### 4.1 Baseline 0 — Logistic Regression (instantaneous)
- **What:** linear classifier on features at time `t` only (no window).
- **Why include it:** the floor; shows what "no temporal info, linear" achieves.
- **Expectation:** collapses as H grows.

### 4.2 Baseline 1 — Random Forest (instantaneous) — *your old "best" model*
- **What:** your existing RF, but predicting `t+H` from features at `t`.
- **Why:** this is the honest reactive baseline. It will look strong at H=1 and weaken with H.
- **Role:** demotes your old headline model to a *baseline*, which is exactly the right framing.

### 4.3 Baseline 2 — Random Forest with **lag features** (flattened window)
- **What:** flatten the `(W × F)` window into one long vector and feed RF.
- **Why critical:** pre-empts the reviewer who says *"you didn't give the tree temporal info."* You did — and the sequence models still win. This is the strongest non-DL competitor.
- **Expectation:** better than instantaneous RF, still below sequence models because flattening discards temporal order/structure.

### 4.4 Model A — 1D-CNN (temporal convolution)
- **What it is:** convolutional filters slide along the **time axis** of the window, detecting local temporal motifs (e.g., "throttle-lift → brake → downshift").
- **Architecture:** `Conv1d → BatchNorm → ReLU` (×2–3 blocks) → global average pool → dense → sigmoid. Input shape `(batch, F, W)`.
- **Why:** cheap, fast, strong baseline for short-range temporal patterns. Captures *local* structure but limited long-range context.
- **Key hyperparams:** kernel size (3–7), channels (32–128), depth (2–4 blocks).

### 4.5 Model B — LSTM / GRU (recurrent)
- **What it is:** processes the window step-by-step, maintaining a hidden "memory" state. **Gates** (input/forget/output for LSTM; reset/update for GRU) let it keep relevant past information and forget noise — solving the vanishing-gradient problem of vanilla RNNs.
- **Why it fits:** telemetry is inherently sequential; the optimal-mode-in-1.5s depends on the *trajectory* through a corner, which recurrence captures naturally.
- **Architecture:** `nn.LSTM(input_size=F, hidden_size=64–128, num_layers=1–2, bidirectional=False)` → take last hidden state → dense → sigmoid. (Use **unidirectional/causal** — you only have the past at inference; bidirectional would leak future and break the real-time premise.)
- **GRU note:** fewer parameters, often as good as LSTM on small data, trains faster — good for a 3-week budget.

```python
class TelemetryLSTM(nn.Module):
    def __init__(self, F, hidden=96, layers=2, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(F, hidden, layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(64, 1))
    def forward(self, x):            # x: (B, W, F)
        out, (h, c) = self.lstm(x)
        return self.head(out[:, -1]) # last timestep -> logit
```

### 4.6 Model C — TCN (Temporal Convolutional Network) — *likely your winner*
- **What it is:** stacked **dilated, causal** 1D convolutions. "Causal" = a prediction at time `t` only uses inputs ≤ `t` (no future leak). "Dilated" = each layer skips exponentially more steps (dilation 1, 2, 4, 8…), so a few layers cover a huge receptive field.
- **Why it often beats LSTM on telemetry:** parallelizable (fast to train on your RTX 3090), stable gradients, long effective memory, no recurrence bottleneck.
- **Architecture:** residual blocks of `[DilatedCausalConv1d → WeightNorm → ReLU → Dropout] ×2` + residual skip; dilation doubles per block; global pool → dense → sigmoid.
- **Hyperparams:** num blocks (4–6), kernel 3, channels 64, dilations `[1,2,4,8,16]`.
- Use the well-known `keras-tcn` or a short PyTorch implementation (Bai et al.-style residual TCN).

### 4.7 Model D — Transformer encoder (self-attention) — *your novelty + XAI hook*
- **What it is:** treats the `W` timesteps as a sequence of tokens; **self-attention** lets each timestep attend to every other timestep and learn *which past moments matter most* for the future prediction (e.g., "the braking point 1.2 s ago is what determines the upcoming straight").
- **Components (explain each in the paper):**
  - **Input projection:** linear map `F → d_model` (e.g., 64/128).
  - **Positional encoding:** since attention is order-agnostic, add sinusoidal or learned position embeddings so the model knows the time order.
  - **Multi-head self-attention:** multiple attention "heads" capture different temporal relations in parallel.
  - **Feed-forward + residual + LayerNorm:** standard encoder block, stack 2–4.
  - **Pooling head:** CLS token or mean-pool → dense → sigmoid.
- **Why it's your best story:** (a) it's the architecture the field expects; (b) **attention weights are directly interpretable** → your XAI section (§9) becomes "which past timesteps drive anticipation," replacing the weak old feature-importance plot.
- **Causality:** apply a causal attention mask so position `i` only attends to `≤ i` (preserves real-time validity).

```python
class TelemetryTransformer(nn.Module):
    def __init__(self, F, d=128, heads=4, layers=3, W=20, p=0.2):
        super().__init__()
        self.proj = nn.Linear(F, d)
        self.pos  = nn.Parameter(torch.randn(1, W, d))   # learned positions
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=256,
                                         dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):                       # (B, W, F)
        z = self.proj(x) + self.pos
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), 1).bool().to(x.device)
        z = self.enc(z, mask=mask)              # causal
        return self.head(z[:, -1])
```

### 4.8 Model ladder summary (what the paper compares)

| Tier | Model | Temporal? | Role |
|---|---|---|---|
| 0 | Logistic Regression (inst.) | No | Floor |
| 1 | Random Forest (inst.) | No | Your old model, now a baseline |
| 2 | Random Forest (lag/flattened) | Weak | Strongest classical competitor |
| A | 1D-CNN | Local | Light DL baseline |
| B | LSTM / GRU | Yes | Recurrent DL |
| C | **TCN** | Yes (long) | Likely best, fast |
| D | **Transformer** | Yes + attention | Novelty + interpretability |

---

## 5. Training setup (Days 4–9)

- **Loss:** `Focal Loss` (handles the ~86/14 imbalance better than `class_weight`), or weighted BCE as fallback. Explain focal loss in the paper (down-weights easy majority samples, focuses on hard transition cases).
- **Optimizer:** `AdamW`, lr `1e-3` with cosine or plateau decay; `weight_decay 1e-4`.
- **Regularization:** dropout 0.2–0.3; early stopping on validation F1; gradient clipping (1.0) for LSTM.
- **Batching:** 256–1024 (windows are small; your 3090 handles this trivially).
- **Reproducibility:** fix seeds, log configs, **release code** (GitHub) — IEEE reviewers value this.
- **Compute reality:** these models are *tiny* (KBs–few MB). Each trains in minutes on a 3090. You can run the full horizon × architecture grid in hours, not days — this is exactly why this plan fits 3 weeks.

---

## 6. Evaluation & metrics (Days 9–11)

**Standard:** F1, Precision/Recall, AUC-ROC, AUC-PR (PR is more honest under imbalance — report it).

**Anticipation-specific (these are your novel, compelling results):**
- **Accuracy/F1 vs horizon H curve** — the centerpiece figure (proves non-circularity + DL advantage).
- **Transition-window F1** — performance specifically on the rare mode-change moments (where value lives). Define a window of ±k steps around each true transition and report metrics there.
- **Lead time** — how many ms before a transition the model reliably flags it.
- **Cross-circuit (LOCO) results** — the generalization table; expect a drop, and expect DL to drop *less* than classical. That gap is a result.

---

## 7. Ablation studies (Days 11–12) — reviewers weight these heavily
Run each as a controlled one-variable change:
1. **Window length W** ∈ {5, 10, 20, 40} — how much past matters.
2. **Horizon H** sweep (already core).
3. **With vs without physics features** (KE, Force, EER) — does domain engineering still help DL?
4. **Architecture** (the ladder in §4.8).
5. **Channel ablation** — drop driver-input channels to test what drives anticipation.
6. **Loss** (focal vs weighted BCE).
7. **(Generalization)** within-circuit vs LOCO.

---

## 8. Statistical testing (Day 12) — reuse your toolkit, now correctly
Your stats were never the problem; the *target* was. Apply the same battery to the real result:
- **DeLong test** on AUC: best DL vs best classical baseline.
- **Paired tests across the LOCO folds / circuits** (Wilcoxon signed-rank) — proper paired comparison, not one giant pooled N.
- **Bootstrap CIs** on F1/AUC per model.
- **Corrected resampled t-test** for CV comparisons (accounts for train-set overlap).
- Avoid the old large-N trap: report **effect sizes and CIs**, not just p-values.

---

## 9. Explainability (Days 12–13) — cheap, high-impact
- **Transformer attention maps:** visualize which past timesteps the model attends to when anticipating a transition (e.g., it locks onto the braking point). One strong figure.
- **SHAP** on the classical/lag-RF baseline for feature attribution (you may already know SHAP from your DS work).
- **Integrated Gradients / saliency** over the input window for the DL models.
This replaces the old MDI feature-importance plot with a temporal-attribution story.

---

## 10. Rewriting the paper (Days 13–17)

You keep most of the document. Map:

| Section | Action |
|---|---|
| Title | Update to "Anticipatory…"; drop "Proof-of-Concept" |
| Abstract | Rewrite around the anticipation task, the horizon curve, DL-vs-classical gap, cross-circuit generalization |
| Intro | Lead with: actuators have latency → reactive prediction is useless → **anticipation is the real problem** → it needs temporal DL. State the non-circular task explicitly. |
| Related work | Reuse your motorsport refs; add 3–4 temporal-DL refs (TCN, Transformer, time-series classification) |
| Dataset | Now multi-circuit; describe LOCO splits |
| Methodology | Replace 4-phase classical pipeline with: windowing + the model ladder + training |
| Results | Horizon curves, model ladder table, LOCO table, transition-window metrics, ablations |
| Discussion | *Why* sequence models win (temporal memory); when classical suffices (small H) |
| Limitations | Honest: still simulated optimal label, limited circuits, no edge benchmark yet |
| Future work | Outcome-based oracle label; edge deployment; multimodal |

**Keep:** K-Means track zones (nice descriptive context) and Isolation Forest (reframe as *exploratory* analysis, not a headline result — and drop the "5% = finding" claim).

---

## 11. IEEE submission mechanics (do this in parallel, Days 1 + 17–20)

- **Template:** `\documentclass[conference]{IEEEtran}`. Use **Overleaf** (search "IEEE Conference Template"). Two-column, 10pt.
- **Page limit:** typically **6 pages** (some allow up to 8 with over-length fee). Confirm *your* conference's exact limit and formatting on its CFP page **today**.
- **References:** IEEE numbered style (`IEEEtran.bst`).
- **Anonymization:** check if the venue is **double-blind**. If so, strip author names/affiliations and anonymize self-references and the GitHub link.
- **IEEE PDF eXpress:** most IEEE conferences require running your final PDF through **PDF eXpress** for compliance — do this 2–3 days before the deadline, not on the 21st.
- **ORCID / copyright (eCF):** be ready to complete the IEEE electronic copyright form on acceptance.
- **Plagiarism/CrossCheck:** IEEE screens similarity — make sure all text is your own and the old draft's reused passages are rewritten to match the new framing.
- **Venue sanity check (critical):** verify the conference is **IEEE-indexed and reputable** (in IEEE Xplore, real program committee, prior proceedings with DOIs). If it looks predatory, redirect your effort.

---

## 12. Day-by-day schedule (31 May → 21 June)

| Days | Dates | Task | Owner split |
|---|---|---|---|
| 1–2 | May 31–Jun 1 | Pull 3–4 circuits via FastF1; rerun preprocessing; set up Overleaf IEEE template | Data person + writer |
| 3 | Jun 2 | Build sliding-window dataset + LOCO splits; sanity checks | Data person |
| 4–5 | Jun 3–4 | Implement baselines (LR, RF inst., RF lag) + CNN; first horizon sweep | ML person |
| 6–7 | Jun 5–6 | LSTM/GRU + TCN; full horizon × architecture grid | ML person |
| 8–9 | Jun 7–8 | Transformer (with causal mask) + tuning; lock model ladder | ML person |
| 9–11 | Jun 8–10 | Metrics: horizon curves, transition-window, **LOCO cross-circuit** | ML + analyst |
| 11–12 | Jun 10–11 | Ablations + statistical tests | analyst |
| 12–13 | Jun 11–12 | XAI: attention maps + SHAP + saliency figures | ML person |
| 13–17 | Jun 12–16 | Rewrite paper; build all final figures/tables | writer + all |
| 17–18 | Jun 16–17 | Internal mock review against §10 rejection list (have your professor read it) | professor + team |
| 18–19 | Jun 17–18 | Revisions; release code repo | all |
| 19–20 | Jun 18–19 | IEEE formatting, references, **PDF eXpress**, anonymization | writer |
| 21 | Jun 20 | Buffer day — final proof, submit early (never submit on the deadline hour) | all |

*One full buffer day before the deadline is non-negotiable — submission portals and PDF eXpress fail at the worst times.*

---

## 13. Scope discipline — what to CUT if you fall behind

Cut from the bottom up; protect the core:
- **Protected core (must have):** anticipation reframe + horizon curve + at least one strong DL model (TCN *or* Transformer) beating the lag-RF baseline + correct stats. *This alone is a valid paper.*
- **Cut first:** Transformer *or* TCN (keep one), then full ablation grid (keep window + horizon + architecture), then multi-circuit (drop to 2 circuits), then XAI (keep just attention maps).
- **Do NOT attempt** (out of scope for 3 weeks): edge/Jetson benchmarking, self-supervised pretraining, multimodal video, an outcome-based physics oracle. List these as future work.

---

## 14. Why this will hold up under review
- The result is **non-circular by construction** (future label, past window).
- DL beats classical for a **principled, explainable** reason (temporal memory + anticipation), not an artifact.
- **Cross-circuit generalization** answers your old single-circuit limitation.
- The **horizon curve** is an unusual, honest figure that signals methodological maturity.
- You **reuse** your real strengths (pipeline, features, rigorous statistics) instead of starting cold.

*Carry-forward principle: a reviewer rewards a modest, honest result on a real problem far more than a spectacular result on a problem solved the moment you defined the label. You're now on the right side of that line.*
