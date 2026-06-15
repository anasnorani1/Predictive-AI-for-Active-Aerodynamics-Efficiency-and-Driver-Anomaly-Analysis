"""
05_xgboost_persistence_loco.py
================================
Two missing baselines evaluated under LOCO at H=10 (1-second horizon):

  XGBoost-instant : XGBClassifier on window[-1]  (last frame, 12 features)
  Persistence     : Predict y[t] as label for y[t+H]  — the naive "no-change" baseline

Why these matter:
  - XGBoost closes the tree-ensemble gap: if XGBoost also wins over deep models,
    the LR result is explained by instantaneous-feature inductive bias, not LR specifically.
  - Persistence gives the trivial lower bound: any model that doesn't beat persistence
    is useless for deployment.

All comparisons use the same LOCO folds and scaled features as 02_baselines.py.

Usage (CPU-only, ~5-10 min):
  cd multi_circuit_work
  python 05_xgboost_persistence_loco.py
  python 05_xgboost_persistence_loco.py --horizons 1 5 10 25 50   # all horizons

Outputs:
  processed/xgb_persistence_results.csv  — metrics per model × fold × horizon
  processed/xgb_preds/                   — {model}_{held_out}_H{H:03d}.npz
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")
    print("         XGBoost rows will be skipped; Persistence will still run.\n")

warnings.filterwarnings("ignore")

WORK_ROOT   = Path(__file__).resolve().parent
PROCESSED   = WORK_ROOT / "processed"
PREDS_DIR   = PROCESSED / "xgb_preds"

# Feature names matching 01_build_anticipatory_dataset.py FEATURE_COLS (order matters)
FEATURE_NAMES = [
    "Speed", "RPM", "nGear", "Throttle", "Brake",
    "X", "Y", "Z",                       # GPS — indices 5,6,7
    "Acceleration", "Elevation_Delta",
    "Kinetic_Energy_MJ", "Longitudinal_Force_N",
]


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    return {
        "n_test"    : len(y_true),
        "n_xmode"   : n_pos,
        "xmode_rate": float(y_true.mean()),
        "f1_macro"  : float(f1_score(y_true, y_pred, average="macro",   zero_division=0)),
        "f1_xmode"  : float(f1_score(y_true, y_pred, pos_label=1,       zero_division=0)),
        "f1_zmode"  : float(f1_score(y_true, y_pred, pos_label=0,       zero_division=0)),
        "auc_roc"   : float(roc_auc_score(y_true, y_prob))            if n_pos > 0 and n_neg > 0 else float("nan"),
        "auc_pr"    : float(average_precision_score(y_true, y_prob))  if n_pos > 0               else float("nan"),
    }


# ── data loading ──────────────────────────────────────────────────────────────

def load_windows(H: int):
    path = PROCESSED / f"windows_H{H:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\nRun 01_build_anticipatory_dataset.py first.")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"].astype(np.int32), d["circuits"], d["lap_keys"]


def available_horizons() -> list[int]:
    return sorted(
        int(p.stem.replace("windows_H", ""))
        for p in PROCESSED.glob("windows_H*.npz")
    )


# ── LOCO loop ─────────────────────────────────────────────────────────────────

def run_loco(H: int) -> list[dict]:
    X, y, circuits, lap_keys = load_windows(H)
    N, W, F = X.shape
    assert F == len(FEATURE_NAMES), f"Feature count mismatch: data has {F}, expected {len(FEATURE_NAMES)}"

    circuit_list = sorted(set(circuits.tolist()))
    rows = []

    for held_out in circuit_list:
        test_mask  = circuits == held_out
        train_mask = ~test_mask
        n_tr = int(train_mask.sum())
        n_te = int(test_mask.sum())
        print(f"  fold={held_out:12s}  train={n_tr:,}  test={n_te:,}")

        y_train = y[train_mask]
        y_test  = y[test_mask]

        # instantaneous features: last frame of each window
        X_inst_tr = X[train_mask, -1, :]   # (n_tr, 12)
        X_inst_te = X[test_mask,  -1, :]   # (n_te, 12)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_inst_tr)
        X_te_sc = scaler.transform(X_inst_te)

        # ── 1. XGBoost-instant ────────────────────────────────────────────────
        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators     = 500,
                max_depth        = 6,
                learning_rate    = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1),
                use_label_encoder= False,
                eval_metric      = "logloss",
                random_state     = 42,
                n_jobs           = -1,
                verbosity        = 0,
            )
            xgb.fit(X_tr_sc, y_train)
            xgb_prob = xgb.predict_proba(X_te_sc)[:, 1]
            xgb_pred = (xgb_prob >= 0.5).astype(int)

            m = metrics(y_test, xgb_prob, xgb_pred)
            m.update({"model": "XGBoost-instant", "held_out": held_out, "H": H})
            rows.append(m)

            np.savez_compressed(
                PREDS_DIR / f"XGBoost-instant_{held_out}_H{H:03d}.npz",
                y_prob=xgb_prob, y_true=y_test, y_pred=xgb_pred,
            )
            print(f"    XGBoost-instant  AUC={m['auc_roc']:.4f}  F1-macro={m['f1_macro']:.4f}  F1-Xmode={m['f1_xmode']:.4f}")

        # ── 2. Persistence baseline ───────────────────────────────────────────
        # y[t] is the current-frame label; we predict it as the label H frames later.
        # The window's last frame gives y[t]; y_test is y[t+H]. So persistence = label of window[-1].
        # The "current label" at prediction time is the mode at the LAST frame of the input window.
        # We read it from Active_Aero_State if available, or recompute from Speed/nGear threshold.

        # Recompute from Speed and nGear in last window frame (indices 0 and 2)
        speed_last = X[test_mask, -1, 0]   # Speed (km/h)
        gear_last  = X[test_mask, -1, 2]   # nGear
        y_persist  = ((speed_last >= 240.0) & (gear_last >= 6)).astype(int)

        # persistence has no "probability" — use 0/1 as prob for AUC
        # (gives AUC ~0.5 if calibration is bad; use actual values for honest reporting)
        persist_prob = y_persist.astype(float)
        m2 = metrics(y_test, persist_prob, y_persist)
        m2.update({"model": "Persistence", "held_out": held_out, "H": H})
        rows.append(m2)

        np.savez_compressed(
            PREDS_DIR / f"Persistence_{held_out}_H{H:03d}.npz",
            y_prob=persist_prob, y_true=y_test, y_pred=y_persist,
        )
        print(f"    Persistence      AUC={m2['auc_roc']:.4f}  F1-macro={m2['f1_macro']:.4f}  F1-Xmode={m2['f1_xmode']:.4f}")

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   help="Horizons to evaluate. Default: all available windows_H*.npz")
    return p.parse_args()


def main():
    args = parse_args()
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    horizons = args.horizons or available_horizons()
    if not horizons:
        raise RuntimeError(f"No windows_H*.npz found in {PROCESSED}. Run 01_build_anticipatory_dataset.py first.")

    print(f"XGBoost + Persistence LOCO  |  horizons={horizons}")
    print(f"XGBoost available: {XGB_AVAILABLE}")
    print("=" * 65)

    all_rows: list[dict] = []

    for H in horizons:
        print(f"\nH={H:3d} (~{H/10:.1f}s ahead)")
        print("-" * 65)
        rows = run_loco(H)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    col_order = ["model", "held_out", "H", "auc_roc", "f1_macro", "f1_xmode", "f1_zmode", "auc_pr", "n_test", "n_xmode", "xmode_rate"]
    df = df[[c for c in col_order if c in df.columns]]

    out_path = PROCESSED / "xgb_persistence_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'=' * 65}")
    print(f"Results saved -> {out_path}")

    # Summary at H=10
    h10 = df[df["H"] == 10]
    if not h10.empty:
        print(f"\n-- Mean across folds at H=10 --")
        print(h10.groupby("model")[["auc_roc", "f1_macro", "f1_xmode"]].mean().round(4).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
