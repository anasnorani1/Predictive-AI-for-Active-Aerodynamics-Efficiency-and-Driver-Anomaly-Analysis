# Execution Order — Anticipatory Aero Pipeline

Run from: `multi_circuit_work/`

---

## Step 0 — One-time setup
```powershell
pip install -r ..\requirements.txt
# For CUDA GPU (if available):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 1 — Pull data + build windowed dataset
```powershell
# First run: fetch all 4 circuits from FastF1 (~10–30 min, needs internet)
python 01_build_anticipatory_dataset.py --pull-missing --w 50 --horizons 1 5 10 25 50

# Paste the stats block (===...===) back to Claude as confirmation.
```
**Output:** `processed/windows_H{H:03d}.npz`, `processed/scaler_holdout_*`, `processed/dataset_stats.txt`

---

## Step 2 — Classical baselines
```powershell
# All horizons, 300 RF trees (~20–40 min on CPU)
python 02_baselines.py --horizons 1 5 10 25 50

# Paste processed/baseline_results.csv summary back to Claude.
```
**Output:** `processed/baseline_results.csv`, `processed/baseline_preds/`

---

## Step 3 — Deep models (primary H=10 first)
```powershell
# Primary result: H=10, all 5 models (~30–90 min on CPU depending on model)
python 03_deep_models.py --horizons 10 --models CNN LSTM GRU TCN Transformer

# Once you have primary results, run the full horizon sweep for top 2 models:
# python 03_deep_models.py --horizons 1 5 10 25 50 --models TCN Transformer
```
**Output:** `processed/deep_results.csv`, `models/*.pt`, `processed/deep_preds/`

---

## Step 4 — Evaluation + stats
```powershell
python 04_evaluate.py --horizon-primary 10

# Paste processed/eval_summary.txt and processed/stats_report.csv back to Claude.
```
**Output:** `graphs/horizon_curve.pdf`, `graphs/loco_heatmap*.pdf`, `processed/table_H010.tex`

---

## Step 5 — XAI (run after Step 3 completes)
```powershell
# Install SHAP first:  pip install shap
python 05_xai.py --H 10 --held-out Monza --model TCN
# Run for all held-out circuits for the paper's best model
```
**Output:** `graphs/ig_*.pdf`, `graphs/shap_*.pdf`, `graphs/attention_*.pdf`

---

## Ablations (after primary results)
```powershell
# Window length ablation
python 01_build_anticipatory_dataset.py --w 20 --horizons 10
python 03_deep_models.py --horizons 10 --models TCN

# Feature ablation: edit FEATURE_COLS in 01_build_anticipatory_dataset.py
# to remove e.g. position features (X,Y,Z) or physics features (KE, Force)
```

---

## Timeline (12 days remaining from 2026-06-09)

| Day | Task |
|-----|------|
| 1 (today) | ✅ Pipeline + all scripts written |
| 1–2 | Data pull finishes; run baselines; paste results |
| 2–3 | Deep models H=10; paste results |
| 3–4 | Full horizon sweep for top 2 models |
| 5 | Evaluation + stats; XAI |
| 6–10 | Paper writing (section by section) |
| 11 | Polish + proofread |
| 12 (Jun 21) | **SUBMIT** |
