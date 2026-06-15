"""Transition-window F1 — low-memory: processes one circuit at a time."""
import gc, pickle, sys, warnings
import numpy as np
import torch
import importlib.util
from pathlib import Path
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("dm", Path(__file__).parent / "03_deep_models.py")
dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)

MODELS_DIR    = Path(__file__).parent / "models"
PROCESSED_DIR = Path(__file__).parent / "processed"
H, W, HALF_WIN = 10, 50, 5
CIRCUITS = ["Monaco", "Monza", "Silverstone", "Suzuka"]

print(f"{'Circuit':12s}  {'Model':12s}  {'N_trans':>8s}  {'F1_trans':>9s}  {'F1_stable':>10s}  {'Delta':>8s}")
print("-" * 70)

rows = []
for model_name in ["Transformer", "GRU"]:
    for held_out in CIRCUITS:
        # Load ONLY the test-circuit slice to avoid OOM
        d = np.load(PROCESSED_DIR / f"windows_H{H:03d}.npz", allow_pickle=True)
        te_mask = d["circuits"] == held_out
        X_te = d["X"][te_mask].copy()
        y_te = d["y"][te_mask].astype(np.int32)
        del d; gc.collect()

        with open(PROCESSED_DIR / f"scaler_holdout_{held_out}_H{H:03d}.pkl", "rb") as f:
            scaler = pickle.load(f)
        N, Ww, Ff = X_te.shape
        X_sc = scaler.transform(X_te.reshape(N, -1)).reshape(N, Ww, Ff).astype(np.float32)
        del X_te; gc.collect()

        # Transition windows
        transitions = np.where(np.diff(y_te) != 0)[0] + 1
        trans_set = set()
        for t in transitions:
            for off in range(-HALF_WIN, HALF_WIN + 1):
                idx = t + off
                if 0 <= idx < len(y_te):
                    trans_set.add(idx)
        trans_idx     = sorted(trans_set)
        non_trans_idx = [i for i in range(len(y_te)) if i not in trans_set]

        ckpt = torch.load(MODELS_DIR / f"{model_name}_fold_{held_out}_H{H:03d}.pt",
                          map_location="cpu", weights_only=False)
        model = dm.build_model(model_name, W, Ff)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(torch.from_numpy(X_sc))).numpy() >= 0.5).astype(np.int32)
        del X_sc, model; gc.collect()

        t_f1 = f1_score(y_te[trans_idx], preds[trans_idx], average="macro", zero_division=0)
        s_f1 = f1_score(y_te[non_trans_idx], preds[non_trans_idx], average="macro", zero_division=0)
        rows.append((held_out, model_name, len(trans_idx), t_f1, s_f1, t_f1 - s_f1))
        print(f"{held_out:12s}  {model_name:12s}  {len(trans_idx):>8d}  {t_f1:>9.4f}  {s_f1:>10.4f}  {t_f1-s_f1:>+8.4f}")

print("-" * 70)
for mn in ["Transformer", "GRU"]:
    r = [x for x in rows if x[1] == mn]
    mt, ms = np.mean([x[3] for x in r]), np.mean([x[4] for x in r])
    print(f"{'MEAN':12s}  {mn:12s}  {'':>8s}  {mt:>9.4f}  {ms:>10.4f}  {mt-ms:>+8.4f}")
