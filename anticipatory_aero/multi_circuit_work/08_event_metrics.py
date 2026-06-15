"""
08_event_metrics.py
====================
Control-relevant EVENT-LEVEL metrics + extra discrimination/calibration metrics.

Reviewer concern (repeated by all three simulated reviewers):
  "AUC and framewise F1 are not the most important outcomes for a controller.
   Report switch-event timing error, early/late actuation rates, excess toggles,
   plus PR-AUC, MCC, and calibration (Brier)."

This script consumes the prediction arrays already saved by 02_baselines.py and
05_xgboost_persistence_loco.py, reconstructs per-lap structure from the windows
file, and computes:

  Frame-level (added):  PR-AUC, MCC, Balanced Accuracy, Brier score
  Event-level (new):    switch-detection rate, mean timing error (frames),
                        early/late actuation split, false-toggle rate

A "switch event" is a Z->X or X->Z transition in y_true within a single lap.
A switch is "detected" if the model's prediction also switches within
+/- TOL frames of the true switch. Timing error is signed (negative = early).

Usage (CPU-only, ~2 min):
  cd multi_circuit_work
  python 08_event_metrics.py
  python 08_event_metrics.py --H 10 --tol 5

Outputs:
  processed/event_metrics_results.csv     — per model x fold
  processed/event_metrics_summary.csv     — mean +/- std across folds
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    f1_score,
)

warnings.filterwarnings("ignore")

WORK_ROOT = Path(__file__).resolve().parent
PROCESSED = WORK_ROOT / "processed"
BASELINE_PREDS = PROCESSED / "baseline_preds"
XGB_PREDS = PROCESSED / "xgb_preds"

CIRCUITS = ["Monaco", "Monza", "Silverstone", "Suzuka"]


# ── per-lap structure reconstruction ──────────────────────────────────────────

def get_test_lapkeys(held_out: str, H: int) -> np.ndarray:
    """
    Reconstruct the lap_keys for the held-out test set, in the SAME order
    used when predictions were generated (boolean mask circuits == held_out
    on the windows file). This matches the row order of the saved y_prob arrays.
    """
    wpath = PROCESSED / f"windows_H{H:03d}.npz"
    d = np.load(wpath, allow_pickle=True)
    circuits = d["circuits"]
    lap_keys = d["lap_keys"]
    mask = circuits == held_out
    return lap_keys[mask]


def find_switches(labels: np.ndarray, lap_keys: np.ndarray) -> list[int]:
    """
    Return indices i where labels[i] != labels[i-1] within the same lap.
    Lap boundaries reset the comparison (no cross-lap switch).
    """
    switches = []
    for lap in np.unique(lap_keys):
        idx = np.where(lap_keys == lap)[0]
        if len(idx) < 2:
            continue
        lab = labels[idx]
        changed = np.where(lab[1:] != lab[:-1])[0] + 1  # local indices
        for c in changed:
            switches.append(idx[c])  # global index of the switch frame
    return sorted(switches)


# ── event-level metrics ───────────────────────────────────────────────────────

def event_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  lap_keys: np.ndarray, tol: int = 5) -> dict:
    """
    Match true switch events to predicted switch events within +/- tol frames.
    """
    true_sw = find_switches(y_true, lap_keys)
    pred_sw = find_switches(y_pred, lap_keys)

    if len(true_sw) == 0:
        return {
            "n_true_switches": 0, "n_pred_switches": len(pred_sw),
            "detection_rate": float("nan"), "mean_timing_err": float("nan"),
            "early_rate": float("nan"), "late_rate": float("nan"),
            "false_toggle_rate": float("nan"),
        }

    pred_sw_arr = np.array(pred_sw)
    matched_pred = set()
    timing_errors = []
    n_detected = 0

    for ts in true_sw:
        if len(pred_sw_arr) == 0:
            continue
        dists = pred_sw_arr - ts            # signed: negative = predicted early
        within = np.where(np.abs(dists) <= tol)[0]
        if len(within) > 0:
            # nearest predicted switch
            nearest = within[np.argmin(np.abs(dists[within]))]
            n_detected += 1
            timing_errors.append(int(dists[nearest]))
            matched_pred.add(int(pred_sw_arr[nearest]))

    timing_errors = np.array(timing_errors) if timing_errors else np.array([0])
    n_false = len(pred_sw) - len(matched_pred)   # predicted switches not matched to any true switch

    return {
        "n_true_switches"  : len(true_sw),
        "n_pred_switches"  : len(pred_sw),
        "detection_rate"   : n_detected / len(true_sw),
        "mean_timing_err"  : float(np.mean(timing_errors)),       # frames; <0 = early
        "early_rate"       : float(np.mean(timing_errors < 0)),
        "late_rate"        : float(np.mean(timing_errors > 0)),
        "false_toggle_rate": n_false / max(len(true_sw), 1),
    }


# ── frame-level extra metrics ─────────────────────────────────────────────────

def frame_metrics(y_true, y_prob, y_pred) -> dict:
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    out = {
        "auc_roc"  : float(roc_auc_score(y_true, y_prob))            if n_pos and n_neg else float("nan"),
        "auc_pr"   : float(average_precision_score(y_true, y_prob))  if n_pos else float("nan"),
        "mcc"      : float(matthews_corrcoef(y_true, y_pred)),
        "bal_acc"  : float(balanced_accuracy_score(y_true, y_pred)),
        "brier"    : float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))) if n_pos and n_neg else float("nan"),
        "f1_macro" : float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return out


# ── model discovery ───────────────────────────────────────────────────────────

def find_pred_file(model: str, held_out: str, H: int) -> Path | None:
    for d in (BASELINE_PREDS, XGB_PREDS):
        f = d / f"{model}_{held_out}_H{H:03d}.npz"
        if f.exists():
            return f
    return None


def discover_models(H: int) -> list[str]:
    models = set()
    for d in (BASELINE_PREDS, XGB_PREDS):
        if not d.exists():
            continue
        for f in d.glob(f"*_H{H:03d}.npz"):
            stem = f.stem.replace(f"_H{H:03d}", "")
            # strip the trailing _<circuit>
            for c in CIRCUITS:
                if stem.endswith(f"_{c}"):
                    models.add(stem[: -(len(c) + 1)])
                    break
    return sorted(models)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--H", type=int, default=10, help="Horizon (default 10)")
    p.add_argument("--tol", type=int, default=5, help="Switch-match tolerance in frames (default 5 = 0.5s)")
    return p.parse_args()


def main():
    args = parse_args()
    H, tol = args.H, args.tol

    models = discover_models(H)
    if not models:
        raise RuntimeError(
            f"No prediction files for H={H} in {BASELINE_PREDS} or {XGB_PREDS}.\n"
            "Run 02_baselines.py and 05_xgboost_persistence_loco.py first."
        )

    print(f"Event-level + extra metrics  |  H={H}  tol=+/-{tol} frames ({tol/10:.1f}s)")
    print(f"Models found: {models}")
    print("=" * 70)

    rows = []
    for model in models:
        for held_out in CIRCUITS:
            f = find_pred_file(model, held_out, H)
            if f is None:
                continue
            d = np.load(f, allow_pickle=True)
            y_true = d["y_true"].astype(int)
            y_prob = d["y_prob"].astype(float)
            y_pred = d["y_pred"].astype(int)

            lap_keys = get_test_lapkeys(held_out, H)
            if len(lap_keys) != len(y_true):
                print(f"  WARN {model} {held_out}: lapkey/pred length mismatch "
                      f"({len(lap_keys)} vs {len(y_true)}) — skipping event metrics")
                ev = {k: float("nan") for k in
                      ["n_true_switches", "n_pred_switches", "detection_rate",
                       "mean_timing_err", "early_rate", "late_rate", "false_toggle_rate"]}
            else:
                ev = event_metrics(y_true, y_pred, lap_keys, tol=tol)

            fm = frame_metrics(y_true, y_prob, y_pred)
            row = {"model": model, "held_out": held_out, "H": H}
            row.update(fm)
            row.update(ev)
            rows.append(row)

            print(f"  {model:18s} {held_out:12s}  "
                  f"PR-AUC={fm['auc_pr']:.3f}  MCC={fm['mcc']:.3f}  Brier={fm['brier']:.3f}  "
                  f"detect={ev['detection_rate'] if not np.isnan(ev['detection_rate']) else float('nan'):.2f}  "
                  f"timing={ev['mean_timing_err']:+.1f}f  "
                  f"falseTog={ev['false_toggle_rate']:.2f}")

    df = pd.DataFrame(rows)
    out_path = PROCESSED / "event_metrics_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-fold results saved -> {out_path}")

    # summary across folds
    agg_cols = ["auc_roc", "auc_pr", "mcc", "bal_acc", "brier",
                "detection_rate", "mean_timing_err", "early_rate",
                "late_rate", "false_toggle_rate"]
    agg_cols = [c for c in agg_cols if c in df.columns]
    summary = df.groupby("model")[agg_cols].agg(["mean", "std"]).round(3)
    sum_path = PROCESSED / "event_metrics_summary.csv"
    summary.to_csv(sum_path)
    print(f"Summary saved        -> {sum_path}")

    print("\n-- Mean across folds (H=10) --")
    mean_only = df.groupby("model")[agg_cols].mean().round(3)
    print(mean_only.to_string())

    print("\nKey:")
    print("  detection_rate   = fraction of true mode switches predicted within +/- tol frames")
    print("  mean_timing_err  = avg frames between true and predicted switch (<0 = predicts early)")
    print("  false_toggle_rate= predicted switches with no matching true switch, per true switch")
    print("  Brier            = calibration error (lower is better)")
    print("\nDone.")


if __name__ == "__main__":
    main()
