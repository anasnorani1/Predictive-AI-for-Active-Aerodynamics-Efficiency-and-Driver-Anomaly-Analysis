"""
02_baselines.py
================
Classical baselines for the anticipatory aerodynamic mode prediction task.

Models trained and evaluated under LOCO (leave-one-circuit-out) for each horizon H:

  LR-instant  : Logistic Regression on window[-1]  (last sample, F features)
  RF-instant  : Random Forest on window[-1]         (last sample, F features)
  RF-lag      : Random Forest on flatten(window)    (W×F features)

RF-lag is the strongest non-deep competitor — if the deep models can't beat it
convincingly, the temporal modelling contribution is weak.

Usage:
  cd multi_circuit_work
  python 02_baselines.py                     # default horizons from windows_H*.npz
  python 02_baselines.py --horizons 1 10 25  # specific horizons
  python 02_baselines.py --fast              # 50 RF trees for a quick smoke-test

Outputs (all under multi_circuit_work/processed/):
  baseline_results.csv          — metrics per model × fold × horizon
  baseline_preds/               — {model}_{held_out}_H{H:03d}.npz (y_prob, y_true, y_pred)
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

WORK_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = WORK_ROOT / "processed"
PREDS_DIR = PROCESSED_DIR / "baseline_preds"
MODELS_DIR = WORK_ROOT / "models"


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    """F1-macro, F1-xmode (class 1), AUC-ROC, AUC-PR."""
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    metrics: dict = {
        "n_test": len(y_true),
        "n_xmode": n_pos,
        "xmode_rate": float(y_true.mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_xmode": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_zmode": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if n_pos > 0 and n_neg > 0 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_prob)) if n_pos > 0 else float("nan"),
    }
    return metrics


def transition_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  lap_keys: np.ndarray) -> float:
    """
    F1 restricted to transition windows — windows where the label changes
    from the previous window within the same lap.  These are the hardest
    and most practically important predictions.
    """
    trans_mask = np.zeros(len(y_true), dtype=bool)
    lap_arr = np.array(lap_keys)
    unique_laps = np.unique(lap_arr)
    for lap in unique_laps:
        idx = np.where(lap_arr == lap)[0]
        if len(idx) < 2:
            continue
        # windows within this lap are assumed contiguous and ordered
        idx_sorted = idx  # build_windows produces them in order
        labels = y_true[idx_sorted]
        changed = np.concatenate([[False], labels[1:] != labels[:-1]])
        trans_mask[idx_sorted[changed]] = True

    if trans_mask.sum() == 0:
        return float("nan")
    return float(f1_score(y_true[trans_mask], y_pred[trans_mask],
                          average="macro", zero_division=0))


# ── helpers ───────────────────────────────────────────────────────────────────

def available_horizons() -> list[int]:
    paths = sorted(PROCESSED_DIR.glob("windows_H*.npz"))
    horizons = []
    for p in paths:
        try:
            h = int(p.stem.replace("windows_H", ""))
            horizons.append(h)
        except ValueError:
            pass
    return horizons


def load_windows(H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / f"windows_H{H:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run 01_build_anticipatory_dataset.py first.")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"].astype(np.int32), data["circuits"], data["lap_keys"]


def load_scaler(held_out: str, H: int) -> StandardScaler | None:
    path = PROCESSED_DIR / f"scaler_holdout_{held_out}_H{H:03d}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   help="Horizons to evaluate. Default: all available windows_H*.npz files.")
    p.add_argument("--fast", action="store_true",
                   help="Use 50 RF trees for a quick smoke-test (default: 300).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    horizons = args.horizons or available_horizons()
    if not horizons:
        raise RuntimeError(f"No windows_H*.npz found in {PROCESSED_DIR}. "
                           "Run 01_build_anticipatory_dataset.py first.")

    n_trees = 50 if args.fast else 300
    print(f"Running baselines  horizons={horizons}  RF n_estimators={n_trees}")

    all_rows: list[dict] = []

    for H in horizons:
        print(f"\n{'=' * 60}")
        print(f"H = {H} (~{H/10:.1f} s ahead)")
        print(f"{'=' * 60}")

        X, y, circuits, lap_keys = load_windows(H)
        N, W, F = X.shape
        circuit_list = sorted(set(circuits.tolist()))

        if len(circuit_list) < 2:
            print(f"  Only 1 circuit ({circuit_list}) — LOCO requires ≥2. Skipping H={H}.")
            continue

        for held_out in circuit_list:
            test_mask = circuits == held_out
            train_mask = ~test_mask
            n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
            print(f"\n  Fold hold-out={held_out}  train={n_tr:,}  test={n_te:,}")

            y_train = y[train_mask]
            y_test = y[test_mask]
            lk_test = lap_keys[test_mask]

            # ── instantaneous features: window[-1] ─────────────────────────
            X_inst_train = X[train_mask, -1, :]   # (n_tr, F)
            X_inst_test = X[test_mask, -1, :]     # (n_te, F)

            scaler_inst = StandardScaler()
            X_inst_train_sc = scaler_inst.fit_transform(X_inst_train)
            X_inst_test_sc = scaler_inst.transform(X_inst_test)

            # ── lag features: flatten window ───────────────────────────────
            scaler_lag = load_scaler(held_out, H)
            if scaler_lag is not None:
                X_lag_train = scaler_lag.transform(X[train_mask].reshape(n_tr, -1))
                X_lag_test = scaler_lag.transform(X[test_mask].reshape(n_te, -1))
            else:
                # fallback: fit scaler on the fly (e.g., only 1 circuit loaded earlier)
                scaler_lag = StandardScaler()
                X_lag_train = scaler_lag.fit_transform(X[train_mask].reshape(n_tr, -1))
                X_lag_test = scaler_lag.transform(X[test_mask].reshape(n_te, -1))

            # ── models ────────────────────────────────────────────────────
            models = {
                "LR-instant": (
                    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                       solver="lbfgs", random_state=42),
                    X_inst_train_sc, X_inst_test_sc,
                ),
                "RF-instant": (
                    RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=42),
                    X_inst_train_sc, X_inst_test_sc,
                ),
                "RF-lag": (
                    RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=42),
                    X_lag_train, X_lag_test,
                ),
            }

            for model_name, (clf, X_tr, X_te) in models.items():
                print(f"    Training {model_name} ...", end=" ", flush=True)
                clf.fit(X_tr, y_train)
                y_prob = clf.predict_proba(X_te)[:, 1]
                y_pred = clf.predict(X_te)

                m = compute_metrics(y_test, y_prob, y_pred)
                t_f1 = transition_f1(y_test, y_pred, lk_test)
                m["f1_transition"] = t_f1
                print(f"F1={m['f1_macro']:.3f}  AUC-ROC={m['auc_roc']:.3f}  AUC-PR={m['auc_pr']:.3f}  "
                      f"F1-trans={t_f1:.3f}")

                row = {"model": model_name, "held_out": held_out, "H": H,
                       "n_train": n_tr, **m}
                all_rows.append(row)

                # save predictions for statistical tests
                np.savez_compressed(
                    PREDS_DIR / f"{model_name}_{held_out}_H{H:03d}.npz",
                    y_prob=y_prob, y_true=y_test, y_pred=y_pred,
                )
                # save RF-lag model for SHAP analysis (05_xai.py)
                if model_name == "RF-lag":
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    with open(MODELS_DIR / f"RF-lag_fold_{held_out}_H{H:03d}_model.pkl", "wb") as fh:
                        pickle.dump(clf, fh, protocol=4)

    # ── save and print results ─────────────────────────────────────────────
    if not all_rows:
        print("\nNo results produced. Check that ≥2 circuit CSVs are loaded.")
        return

    results = pd.DataFrame(all_rows)
    out_path = PROCESSED_DIR / "baseline_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    # summary table: mean ± std across LOCO folds
    print("\n── Baseline summary (mean ± std across LOCO folds) ─────────────────")
    summary = (
        results
        .groupby(["model", "H"])[["f1_macro", "f1_xmode", "auc_roc", "auc_pr", "f1_transition"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    print(summary.to_string())


if __name__ == "__main__":
    main()
