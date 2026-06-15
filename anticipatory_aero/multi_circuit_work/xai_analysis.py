"""Run after 05_xai.py — computes transition-window F1 and IG feature rankings."""
import numpy as np
import pickle
import torch
import importlib.util
import sys
import warnings
from pathlib import Path
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("dm", Path(__file__).parent / "03_deep_models.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

MODELS_DIR    = Path(__file__).parent / "models"
PROCESSED_DIR = Path(__file__).parent / "processed"
H, W, HALF_WIN = 10, 50, 5
CIRCUITS = ["Monaco", "Monza", "Silverstone", "Suzuka"]
FEATURE_COLS = [
    "Speed", "RPM", "nGear", "Throttle", "Brake", "X", "Y", "Z",
    "Acceleration", "Elevation_Delta", "Kinetic_Energy_MJ", "Longitudinal_Force_N",
]


def get_windows(held_out):
    d = np.load(PROCESSED_DIR / f"windows_H{H:03d}.npz", allow_pickle=True)
    X, y, circs = d["X"], d["y"].astype(np.int32), d["circuits"]
    te_mask = circs == held_out
    X_te, y_te = X[te_mask], y[te_mask]
    with open(PROCESSED_DIR / f"scaler_holdout_{held_out}_H{H:03d}.pkl", "rb") as f:
        scaler = pickle.load(f)
    N, Ww, Ff = X_te.shape
    X_sc = scaler.transform(X_te.reshape(N, -1)).reshape(N, Ww, Ff).astype(np.float32)
    return X_sc, y_te


def predict(model_name, held_out, X_sc):
    ckpt = torch.load(
        MODELS_DIR / f"{model_name}_fold_{held_out}_H{H:03d}.pt",
        map_location="cpu", weights_only=False,
    )
    _, Ww, Ff = X_sc.shape
    model = dm.build_model(model_name, Ww, Ff)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_sc))).numpy()
    return (probs >= 0.5).astype(np.int32), probs


def transition_indices(y_te, half_win=HALF_WIN):
    transitions = np.where(np.diff(y_te) != 0)[0] + 1
    trans_set = set()
    for t in transitions:
        for offset in range(-half_win, half_win + 1):
            idx = t + offset
            if 0 <= idx < len(y_te):
                trans_set.add(idx)
    trans_idx = sorted(trans_set)
    non_trans_idx = [i for i in range(len(y_te)) if i not in trans_set]
    return trans_idx, non_trans_idx


# ── Transition-window F1 ──────────────────────────────────────────────────────
print("=" * 70)
print("TRANSITION-WINDOW F1  (+-5 steps = +-0.5s around each mode change)")
print("=" * 70)
print(f"{'Circuit':12s}  {'Model':12s}  {'N_trans':>8s}  {'F1_trans':>9s}  {'F1_stable':>10s}  {'Delta':>8s}")
print("-" * 70)

rows = []
for model_name in ["Transformer", "GRU"]:
    for held_out in CIRCUITS:
        X_sc, y_te = get_windows(held_out)
        preds, _ = predict(model_name, held_out, X_sc)
        trans_idx, non_trans_idx = transition_indices(y_te)
        t_f1 = f1_score(y_te[trans_idx], preds[trans_idx], average="macro", zero_division=0)
        s_f1 = f1_score(y_te[non_trans_idx], preds[non_trans_idx], average="macro", zero_division=0)
        delta = t_f1 - s_f1
        rows.append((held_out, model_name, len(trans_idx), t_f1, s_f1, delta))
        print(f"{held_out:12s}  {model_name:12s}  {len(trans_idx):>8d}  {t_f1:>9.4f}  {s_f1:>10.4f}  {delta:>+8.4f}")

# Mean over circuits per model
print("-" * 70)
for model_name in ["Transformer", "GRU"]:
    r = [x for x in rows if x[1] == model_name]
    mean_t = np.mean([x[3] for x in r])
    mean_s = np.mean([x[4] for x in r])
    print(f"{'MEAN':12s}  {model_name:12s}  {'':>8s}  {mean_t:>9.4f}  {mean_s:>10.4f}  {mean_t-mean_s:>+8.4f}")

# ── IG feature importance ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("IG FEATURE IMPORTANCE  (mean |IG| per feature, averaged over time)")
print("=" * 70)

for model_name in ["GRU", "Transformer"]:
    print(f"\n--- {model_name} ---")
    print(f"{'Circuit':12s}  {'Rank1':18s}  {'Rank2':18s}  {'Rank3':18s}")
    for held_out in CIRCUITS:
        p = PROCESSED_DIR / f"ig_{model_name}_{held_out}_H{H:03d}.npy"
        if not p.exists():
            print(f"{held_out:12s}  [missing]")
            continue
        arr = np.load(p)           # (W, F)
        feat_mean = arr.mean(axis=0)
        order = np.argsort(feat_mean)[::-1]
        top = [f"{FEATURE_COLS[i]} ({feat_mean[i]:.4f})" for i in order[:3]]
        print(f"{held_out:12s}  {top[0]:18s}  {top[1]:18s}  {top[2]:18s}")

    # Also print normalised full ranking for Monaco
    p = PROCESSED_DIR / f"ig_{model_name}_Monaco_H{H:03d}.npy"
    if p.exists():
        arr = np.load(p)
        feat_mean = arr.mean(axis=0)
        total = feat_mean.sum()
        order = np.argsort(feat_mean)[::-1]
        print(f"\n  {model_name} Monaco full ranking (% of total |IG|):")
        for rank, i in enumerate(order):
            pct = 100 * feat_mean[i] / total if total > 0 else 0
            print(f"    {rank+1:2d}. {FEATURE_COLS[i]:25s} {pct:5.1f}%")

# ── Per-time-step importance (which time lag matters most) ───────────────────
print()
print("=" * 70)
print("IG TIME-STEP IMPORTANCE  (mean |IG| per time step, Monaco)")
print("=" * 70)

for model_name in ["GRU", "Transformer"]:
    p = PROCESSED_DIR / f"ig_{model_name}_Monaco_H{H:03d}.npy"
    if not p.exists():
        continue
    arr = np.load(p)       # (W, F)
    time_mean = arr.mean(axis=1)  # (W,) — importance per time step
    # Summarise as early / mid / late thirds
    early = time_mean[:16].mean()
    mid   = time_mean[16:33].mean()
    late  = time_mean[33:].mean()
    peak  = np.argmax(time_mean)
    print(f"{model_name:12s} Monaco:  early(0-15)={early:.5f}  mid(16-32)={mid:.5f}  "
          f"late(33-49)={late:.5f}  peak_step={peak}")
