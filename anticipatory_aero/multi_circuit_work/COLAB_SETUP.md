# Colab Training Guide

Total time on T4 GPU: ~20–40 minutes for all 5 models at H=10.

---

## Step 1 — Open Colab and set runtime

1. Go to https://colab.research.google.com → New notebook
2. Runtime → Change runtime type → **T4 GPU** → Save

---

## Step 2 — Upload files

Click the folder icon (left sidebar) → Upload button.  
Upload ALL of these files from `multi_circuit_work/processed/`:

**Windows (5 files, ~40 MB total):**
```
windows_H001.npz
windows_H005.npz
windows_H010.npz
windows_H025.npz
windows_H050.npz
```

**Scalers (20 files):**
```
scaler_holdout_Monaco_H001.pkl
scaler_holdout_Monaco_H005.pkl
scaler_holdout_Monaco_H010.pkl
scaler_holdout_Monaco_H025.pkl
scaler_holdout_Monaco_H050.pkl
scaler_holdout_Monza_H001.pkl
scaler_holdout_Monza_H005.pkl
scaler_holdout_Monza_H010.pkl
scaler_holdout_Monza_H025.pkl
scaler_holdout_Monza_H050.pkl
scaler_holdout_Silverstone_H001.pkl
scaler_holdout_Silverstone_H005.pkl
scaler_holdout_Silverstone_H010.pkl
scaler_holdout_Silverstone_H025.pkl
scaler_holdout_Silverstone_H050.pkl
scaler_holdout_Suzuka_H001.pkl
scaler_holdout_Suzuka_H005.pkl
scaler_holdout_Suzuka_H010.pkl
scaler_holdout_Suzuka_H025.pkl
scaler_holdout_Suzuka_H050.pkl
```

**Script:**
```
03_deep_models.py
```

All files land in `/content/` on Colab.

---

## Step 3 — Install dependencies

Paste into a cell and run:
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
!pip install scikit-learn scipy numpy pandas -q
import torch
print("CUDA:", torch.cuda.is_available(), "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## Step 4 — Run primary training (H=10, all 5 models, all 4 LOCO folds)

```python
# Create output directories  
import os
os.makedirs('/content/deep_preds', exist_ok=True)
os.makedirs('/content/models', exist_ok=True)

# Run training
!python /content/03_deep_models.py \
    --horizons 10 \
    --models CNN LSTM GRU TCN Transformer \
    --max-epochs 80 \
    --patience 12 \
    --data-dir /content \
    --out-dir /content
```

Expected output per fold:
```
[CNN] ... ep=XX  F1=0.XXX  AUC-ROC=0.XXX  AUC-PR=0.XXX
[LSTM] ... ep=XX ...
...
```

---

## Step 5 — Run horizon sweep (best 2 models, all 5 H values)

After seeing Step 4 results, run the top 2 performing models across all horizons:
```python
# Replace MODEL1 and MODEL2 with the two best from Step 4
!python /content/03_deep_models.py \
    --horizons 1 5 10 25 50 \
    --models TCN Transformer \
    --max-epochs 80 \
    --patience 12 \
    --data-dir /content \
    --out-dir /content
```

---

## Step 6 — Download results

```python
# Download the results CSV and predictions
from google.colab import files

files.download('/content/deep_results.csv')

# Also download model checkpoints (large — do this after downloading CSV first)
# import shutil
# shutil.make_archive('/content/deep_models', 'zip', '/content/models')
# files.download('/content/deep_models.zip')
```

---

## Step 7 — Paste back to Claude

Download `deep_results.csv` and paste its contents back. That's all I need to:
1. Fill in all the [FILL] placeholders in the paper
2. Run statistical tests
3. Generate all figures

---

## What to expect (benchmark numbers, NOT your results)

At H=10, typical LOCO results on multi-circuit racing telemetry:
- RF-instant: AUC-ROC ~0.93 (your early results confirm this)
- LSTM: AUC-ROC ~0.95+ with proper training (already 0.952 at 10 epochs)
- TCN/Transformer: expect similar or better than LSTM
- Monaco fold will be hardest (~0.85-0.95 AUC for deep models vs 0.90 for LR)

If deep models beat RF-lag across all circuits and particularly on Monaco:
→ This is the headline result: **temporal learned representations generalize across speed profiles where lag-augmented trees fail**

If deep models don't beat baselines:
→ We reframe: **the task is surprisingly solved by instantaneous classifiers at H≤10s; we characterize the anticipatory difficulty and show where temporal context matters (H≥25s)**

Either finding is publishable. Don't overclaim.
