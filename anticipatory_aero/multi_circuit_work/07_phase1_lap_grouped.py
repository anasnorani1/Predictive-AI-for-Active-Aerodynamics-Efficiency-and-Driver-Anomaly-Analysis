"""
07_phase1_lap_grouped.py
=========================
Re-runs Phase 1 classification (LR + RF) using a LAP-GROUPED 80/20 split
instead of the random frame-level split used in the original paper.

Reviewer concern: at 10 Hz, adjacent frames are highly correlated.
A random split leaks near-duplicate samples across train/test, potentially
inflating AUC. Grouping by lap ensures no lap appears in both train and test.

Uses the Suzuka 2026 data (Phase 1 only) from anomaly_detection/artefacts/.

Expected result: RF AUC should remain 1.000 (the rule is deterministic —
any split exposes the same threshold boundary). LR AUC should be ≥0.965.
If both hold, the original Phase 1 result is valid regardless of split method.

Usage (CPU-only, ~1 min):
  cd multi_circuit_work
  python 07_phase1_lap_grouped.py

Outputs:
  processed/phase1_lap_grouped_results.csv   — grouped vs random split comparison
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

WORK_ROOT      = Path(__file__).resolve().parent
PROCESSED      = WORK_ROOT / "processed"
ANOMALY_ART    = Path(__file__).resolve().parents[2] / "anomaly_detection" / "artefacts"

RANDOM_SEED    = 42
TEST_SIZE      = 0.20


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, y_pred) -> dict:
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    return {
        "accuracy" : float((y_pred == y_true).mean()),
        "f1_macro" : float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_xmode" : float(f1_score(y_true, y_pred, pos_label=1,     zero_division=0)),
        "f1_zmode" : float(f1_score(y_true, y_pred, pos_label=0,     zero_division=0)),
        "auc_roc"  : float(roc_auc_score(y_true, y_prob))          if n_pos > 0 and n_neg > 0 else float("nan"),
        "auc_pr"   : float(average_precision_score(y_true, y_prob)) if n_pos > 0               else float("nan"),
        "n_test"   : len(y_true),
        "n_xmode"  : n_pos,
    }


# ── load Phase 1 data ─────────────────────────────────────────────────────────

def load_phase1_data():
    """
    Load the Suzuka Phase 1 dataset from anomaly_detection/artefacts/.
    Falls back to the multi_circuit_work preprocessed CSV (Suzuka only).
    """
    # Option A: pre-built train/test arrays
    X_train_path = ANOMALY_ART / "X_train.npy"
    X_test_path  = ANOMALY_ART / "X_test.npy"
    y_train_path = ANOMALY_ART / "y_train.npy"
    y_test_path  = ANOMALY_ART / "y_test.npy"
    df_path      = ANOMALY_ART / "df_preprocessed.csv"

    if not df_path.exists():
        # Try reading from multi_circuit_work processed data (Suzuka only)
        mc_path = PROCESSED / "multi_circuit_preprocessed.csv"
        if not mc_path.exists():
            raise FileNotFoundError(
                f"Cannot find Phase 1 data at {df_path} or {mc_path}.\n"
                "Ensure anomaly_detection/artefacts/ contains df_preprocessed.csv."
            )
        print(f"Using multi-circuit data filtered to Suzuka: {mc_path}")
        df = pd.read_csv(mc_path)
        df = df[df["Circuit"] == "Suzuka"].copy()
        return df, None

    print(f"Loading Phase 1 data from {df_path}")
    df = pd.read_csv(df_path)
    return df, (X_train_path, X_test_path, y_train_path, y_test_path)


def get_feature_cols():
    feat_path = ANOMALY_ART / "feature_cols.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as fh:
            return pickle.load(fh)
    # Fallback: same columns as used in Phase 2 windows
    return [
        "Speed", "RPM", "nGear", "Throttle", "Brake",
        "Acceleration", "Elevation_Delta",
        "Kinetic_Energy_MJ", "Longitudinal_Force_N",
    ]


# ── split helpers ─────────────────────────────────────────────────────────────

def random_frame_split(X, y, seed=RANDOM_SEED, test_size=TEST_SIZE):
    """Reproduce the original random 80/20 frame-level split."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    tr_idx = idx[:split]
    te_idx = idx[split:]
    return tr_idx, te_idx


def lap_grouped_split(df_full, feat_cols, target_col, lap_col="LapNumber",
                      test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Group by LapNumber so no lap appears in both train and test.
    Returns (X_train, X_test, y_train, y_test, n_train_laps, n_test_laps).
    """
    groups = df_full[lap_col].values
    X = df_full[feat_cols].values.astype(np.float32)
    y = df_full[target_col].values.astype(np.int32)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    n_train_laps = len(set(groups[tr_idx]))
    n_test_laps  = len(set(groups[te_idx]))

    return (X[tr_idx], X[te_idx],
            y[tr_idx], y[te_idx],
            n_train_laps, n_test_laps)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Phase 1 — Lap-Grouped Split Validation")
    print("=" * 65)

    df, arrays = load_phase1_data()
    feat_cols = get_feature_cols()

    # Keep only columns that exist
    feat_cols = [c for c in feat_cols if c in df.columns]
    target_col = "Active_Aero_State" if "Active_Aero_State" in df.columns else "Optimal_Aero"
    lap_col    = "LapNumber" if "LapNumber" in df.columns else None

    print(f"Dataset shape   : {df.shape}")
    print(f"Feature columns : {feat_cols}")
    print(f"Target column   : {target_col}")
    print(f"Lap column      : {lap_col}")
    print(f"X-Mode fraction : {df[target_col].mean():.3f}")
    print()

    X_all = df[feat_cols].values.astype(np.float32)
    y_all = df[target_col].values.astype(np.int32)

    models = {
        "LR": LogisticRegression(C=1.0, max_iter=2000,
                                 class_weight="balanced",
                                 solver="lbfgs", random_state=RANDOM_SEED),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=15,
                                     class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_SEED),
    }

    all_rows = []

    # -- Split A: random frame-level (reproduce original) -------------------
    print("-- Condition A: Random frame-level split (original paper) --")
    tr_idx, te_idx = random_frame_split(X_all, y_all)
    X_tr_r, X_te_r = X_all[tr_idx], X_all[te_idx]
    y_tr_r, y_te_r = y_all[tr_idx], y_all[te_idx]

    scaler_r = StandardScaler()
    X_tr_r_sc = scaler_r.fit_transform(X_tr_r)
    X_te_r_sc = scaler_r.transform(X_te_r)

    print(f"  Train: {len(tr_idx):,} frames   Test: {len(te_idx):,} frames")

    for model_name, clf in models.items():
        clf.fit(X_tr_r_sc, y_tr_r)
        y_prob = clf.predict_proba(X_te_r_sc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_metrics(y_te_r, y_prob, y_pred)
        m.update({"model": model_name, "split": "random-frame", "n_groups": "N/A"})
        all_rows.append(m)
        print(f"  {model_name:3s}  AUC={m['auc_roc']:.4f}  F1-macro={m['f1_macro']:.4f}  "
              f"F1-Xmode={m['f1_xmode']:.4f}  Accuracy={m['accuracy']:.4f}")

    # -- Split B: lap-grouped ------------------------------------------------
    print()
    if lap_col is None:
        print("WARNING: LapNumber column not found - skipping lap-grouped split.")
        print("         Ensure df_preprocessed.csv includes LapNumber.")
    else:
        print("-- Condition B: Lap-grouped split (grouped by LapNumber) --")
        (X_tr_g, X_te_g, y_tr_g, y_te_g,
         n_tr_laps, n_te_laps) = lap_grouped_split(df, feat_cols, target_col, lap_col)

        scaler_g = StandardScaler()
        X_tr_g_sc = scaler_g.fit_transform(X_tr_g)
        X_te_g_sc = scaler_g.transform(X_te_g)

        print(f"  Train: {len(y_tr_g):,} frames ({n_tr_laps} laps)   "
              f"Test: {len(y_te_g):,} frames ({n_te_laps} laps)")

        for model_name, clf in models.items():
            # Re-instantiate to avoid contamination from Condition A
            clf_fresh = clf.__class__(**clf.get_params())
            clf_fresh.fit(X_tr_g_sc, y_tr_g)
            y_prob = clf_fresh.predict_proba(X_te_g_sc)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            m = compute_metrics(y_te_g, y_prob, y_pred)
            m.update({"model": model_name, "split": "lap-grouped",
                      "n_groups": f"{n_tr_laps}+{n_te_laps} laps"})
            all_rows.append(m)
            print(f"  {model_name:3s}  AUC={m['auc_roc']:.4f}  F1-macro={m['f1_macro']:.4f}  "
                  f"F1-Xmode={m['f1_xmode']:.4f}  Accuracy={m['accuracy']:.4f}")

    # -- Save ----------------------------------------------------------------
    df_out = pd.DataFrame(all_rows)
    col_order = ["model", "split", "n_groups", "auc_roc", "f1_macro", "f1_xmode",
                 "f1_zmode", "auc_pr", "accuracy", "n_test", "n_xmode"]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    out_path = PROCESSED / "phase1_lap_grouped_results.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n{'=' * 65}")
    print(f"Results saved -> {out_path}")

    # Comparison delta
    if "lap-grouped" in df_out["split"].values:
        print("\n-- AUC delta (random - grouped) --")
        for model in ["LR", "RF"]:
            rows_m = df_out[df_out["model"] == model]
            if len(rows_m) == 2:
                auc_rand  = rows_m[rows_m["split"] == "random-frame"]["auc_roc"].values[0]
                auc_group = rows_m[rows_m["split"] == "lap-grouped"]["auc_roc"].values[0]
                delta = auc_rand - auc_group
                verdict = "negligible" if abs(delta) < 0.005 else "SIGNIFICANT"
                print(f"  {model:3s}  random={auc_rand:.4f}  grouped={auc_group:.4f}  "
                      f"delta={delta:+.4f}  [{verdict}]")

    print("\nInterpretation:")
    print("  |delta| < 0.005 -> lap-level grouping has no material effect on AUC")
    print("                  -> original random-split result is not inflated by temporal leakage")
    print("  |delta| > 0.010 -> temporal leakage was present; report grouped-split AUC in paper")
    print("\nDone.")


if __name__ == "__main__":
    main()
