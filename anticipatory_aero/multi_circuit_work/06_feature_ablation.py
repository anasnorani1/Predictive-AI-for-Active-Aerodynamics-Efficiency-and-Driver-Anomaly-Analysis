"""
06_feature_ablation.py
=======================
Ablation study: does removing GPS/position features change results?

Reviewer concern: features X, Y, Z encode circuit identity. If models rely on
absolute position to predict the label, results may not generalise to unseen
circuits via the physics alone — the model is partly memorising circuit layout.

This script runs LR-instant and RF-instant under LOCO in two conditions:

  FULL    : all 12 features (matches 02_baselines.py exactly)
  NO-POS  : 9 features — X, Y, Z removed (indices 5, 6, 7 removed)

If AUC is stable across conditions → results are physics-driven, not position-memorisation.
If AUC drops significantly in NO-POS → position leakage is contributing.

Usage (CPU-only, ~3-5 min):
  cd multi_circuit_work
  python 06_feature_ablation.py
  python 06_feature_ablation.py --horizon 10          # H=10 only (default)
  python 06_feature_ablation.py --horizon 10 25 50    # multiple horizons

Outputs:
  processed/feature_ablation_results.csv   — full vs no-pos comparison table
  processed/ablation_preds/                — prediction arrays per condition/fold
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

WORK_ROOT = Path(__file__).resolve().parent
PROCESSED = WORK_ROOT / "processed"
PREDS_DIR = PROCESSED / "ablation_preds"

# Feature names in the same order as 01_build_anticipatory_dataset.py
FEATURE_NAMES = [
    "Speed",               # 0
    "RPM",                 # 1
    "nGear",               # 2
    "Throttle",            # 3
    "Brake",               # 4
    "X",                   # 5  ← GPS position (circuit identity leakage candidate)
    "Y",                   # 6  ← GPS position
    "Z",                   # 7  ← GPS altitude
    "Acceleration",        # 8
    "Elevation_Delta",     # 9
    "Kinetic_Energy_MJ",   # 10
    "Longitudinal_Force_N",# 11
]

# Indices to remove in the NO-POS condition
POSITION_INDICES = [5, 6, 7]    # X, Y, Z
PHYSICS_INDICES  = [i for i in range(len(FEATURE_NAMES)) if i not in POSITION_INDICES]

PHYSICS_FEATURE_NAMES = [FEATURE_NAMES[i] for i in PHYSICS_INDICES]


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, y_pred) -> dict:
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    return {
        "f1_macro" : float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_xmode" : float(f1_score(y_true, y_pred, pos_label=1,     zero_division=0)),
        "f1_zmode" : float(f1_score(y_true, y_pred, pos_label=0,     zero_division=0)),
        "auc_roc"  : float(roc_auc_score(y_true, y_prob))          if n_pos > 0 and n_neg > 0 else float("nan"),
        "auc_pr"   : float(average_precision_score(y_true, y_prob)) if n_pos > 0               else float("nan"),
        "n_test"   : len(y_true),
        "n_xmode"  : n_pos,
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

def run_ablation(H: int, n_trees: int = 300) -> list[dict]:
    X, y, circuits, lap_keys = load_windows(H)
    N, W, F = X.shape

    assert F == 12, f"Expected 12 features, got {F}. Check FEATURE_NAMES order."

    circuit_list = sorted(set(circuits.tolist()))
    rows = []

    conditions = {
        "FULL"  : list(range(12)),     # all features
        "NO-POS": PHYSICS_INDICES,     # X, Y, Z removed
    }

    for held_out in circuit_list:
        test_mask  = circuits == held_out
        train_mask = ~test_mask
        n_tr = int(train_mask.sum())
        n_te = int(test_mask.sum())
        print(f"\n  fold={held_out:12s}  train={n_tr:,}  test={n_te:,}")

        y_train = y[train_mask]
        y_test  = y[test_mask]

        for condition, feat_idx in conditions.items():
            feat_label = f"[{len(feat_idx)} features]"
            # Use last frame of each window (instantaneous)
            X_tr = X[train_mask, -1, :][:, feat_idx]
            X_te = X[test_mask,  -1, :][:, feat_idx]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            for model_name, clf in [
                ("LR-instant",
                 LogisticRegression(C=1.0, max_iter=2000,
                                    class_weight="balanced",
                                    solver="lbfgs", random_state=42)),
                ("RF-instant",
                 RandomForestClassifier(n_estimators=n_trees, max_depth=15,
                                        class_weight="balanced",
                                        n_jobs=-1, random_state=42)),
            ]:
                clf.fit(X_tr_sc, y_train)
                y_prob = clf.predict_proba(X_te_sc)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                m = compute_metrics(y_test, y_prob, y_pred)
                m.update({
                    "model"    : model_name,
                    "condition": condition,
                    "features" : feat_label,
                    "held_out" : held_out,
                    "H"        : H,
                })
                rows.append(m)

                np.savez_compressed(
                    PREDS_DIR / f"{model_name}_{condition}_{held_out}_H{H:03d}.npz",
                    y_prob=y_prob, y_true=y_test, y_pred=y_pred,
                )
                print(f"    {model_name:12s} {condition:7s} {feat_label}  "
                      f"AUC={m['auc_roc']:.4f}  F1-macro={m['f1_macro']:.4f}  F1-Xmode={m['f1_xmode']:.4f}")

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--horizon", type=int, nargs="+", default=[10],
                   help="Horizons to evaluate (default: 10)")
    p.add_argument("--fast", action="store_true",
                   help="Use 50 RF trees for quick smoke-test")
    return p.parse_args()


def main():
    args = parse_args()
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    horizons  = args.horizon
    n_trees   = 50 if args.fast else 300

    print("Feature Ablation: FULL vs NO-POS (X, Y, Z removed)")
    print(f"Position features removed: {[FEATURE_NAMES[i] for i in POSITION_INDICES]}")
    print(f"Remaining physics features: {PHYSICS_FEATURE_NAMES}")
    print(f"Horizons: {horizons}   RF n_estimators: {n_trees}")
    print("=" * 65)

    all_rows: list[dict] = []
    for H in horizons:
        print(f"\nH={H:3d} (~{H/10:.1f}s ahead)")
        print("-" * 65)
        rows = run_ablation(H, n_trees=n_trees)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    col_order = ["model", "condition", "features", "held_out", "H",
                 "auc_roc", "f1_macro", "f1_xmode", "f1_zmode", "auc_pr",
                 "n_test", "n_xmode"]
    df = df[[c for c in col_order if c in df.columns]]

    out_path = PROCESSED / "feature_ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'=' * 65}")
    print(f"Results saved -> {out_path}")

    # Pivot comparison table
    h10 = df[df["H"] == 10]
    if not h10.empty:
        print("\n-- Mean AUC-ROC across folds (H=10) --")
        pivot = h10.groupby(["model", "condition"])["auc_roc"].mean().unstack("condition").round(4)
        if "FULL" in pivot.columns and "NO-POS" in pivot.columns:
            pivot["AUC_drop"] = (pivot["FULL"] - pivot["NO-POS"]).round(4)
        print(pivot.to_string())
        print("\nInterpretation:")
        print("  AUC_drop ~ 0.000-0.005 -> position features are redundant (physics-driven result)")
        print("  AUC_drop > 0.010       -> GPS features carry circuit-specific information")

    print("\nDone.")


if __name__ == "__main__":
    main()
