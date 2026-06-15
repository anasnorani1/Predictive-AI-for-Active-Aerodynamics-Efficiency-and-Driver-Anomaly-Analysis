"""
09_physics_baseline.py
======================
Physics threshold-crossing baselines for the anticipatory proxy task.

Reviewer point (deep-research re-review): for a target that is a deterministic
threshold on FUTURE speed and gear, the most natural non-ML comparator is a
kinematic extrapolation of speed/gear followed by the threshold rule. If such a
baseline nearly matches Logistic Regression, the "simple beats deep" story is
even stronger; if it beats LR, the story changes. Either way it must be reported.

Two parameter-free predictors (no training):
  Physics-CV : constant-velocity   -> future_speed = speed_t ; future_gear = gear_t
               (this is equivalent to applying the rule to the current frame)
  Physics-CA : constant-accel       -> future_speed = speed_t + accel_t * H
               future_gear = gear_t  (gear persistence)
  prediction : y_hat = 1[ future_speed >= 240 AND future_gear >= 6 ]

Both are evaluated directly on each circuit's H=10 windows (no LOCO training,
since there are no parameters); we report the per-circuit mean to align with the
LOCO test sets used by the learned models.

Usage (CPU, <1 min):  python 09_physics_baseline.py
Output: processed/physics_baseline_results.csv
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
SPEED_THR, GEAR_THR, H = 240.0, 6, 10
CIRCUITS = ["Monaco", "Monza", "Silverstone", "Suzuka"]
# feature indices in the windows (see 01_build_anticipatory_dataset.py)
I_SPEED, I_GEAR, I_ACC = 0, 2, 8


def soft_score(future_speed, future_gear):
    """Smooth confidence for AUC: logistic on speed margin, gated by gear eligibility."""
    s = 1.0 / (1.0 + np.exp(-(future_speed - SPEED_THR) / 5.0))
    return s * (future_gear >= GEAR_THR)


def metrics(yt, score, yp):
    npos, nneg = int(yt.sum()), int((1 - yt).sum())
    return dict(
        auc_roc=float(roc_auc_score(yt, score)) if npos and nneg else float("nan"),
        auc_pr=float(average_precision_score(yt, score)) if npos else float("nan"),
        f1_macro=float(f1_score(yt, yp, average="macro", zero_division=0)),
        f1_xmode=float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        mcc=float(matthews_corrcoef(yt, yp)) if npos and nneg else float("nan"),
        accuracy=float((yp == yt).mean()),
    )


def main():
    w = np.load(PROC / f"windows_H{H:03d}.npz", allow_pickle=True)
    X, y, circ = w["X"], w["y"].astype(int), w["circuits"]
    rows = []
    for c in CIRCUITS:
        m = circ == c
        sp = X[m, -1, I_SPEED]; gr = X[m, -1, I_GEAR]; ac = X[m, -1, I_ACC]
        yt = y[m]
        # Physics-CV (constant velocity)
        fs_cv, fg_cv = sp, gr
        yp_cv = ((fs_cv >= SPEED_THR) & (fg_cv >= GEAR_THR)).astype(int)
        mc = metrics(yt, soft_score(fs_cv, fg_cv), yp_cv); mc.update(model="Physics-CV", held_out=c)
        rows.append(mc)
        # Physics-CA (constant acceleration extrapolation of speed)
        fs_ca = sp + ac * H; fg_ca = gr
        yp_ca = ((fs_ca >= SPEED_THR) & (fg_ca >= GEAR_THR)).astype(int)
        ma = metrics(yt, soft_score(fs_ca, fg_ca), yp_ca); ma.update(model="Physics-CA", held_out=c)
        rows.append(ma)
        print(f"{c:12s}  CV: AUC={mc['auc_roc']:.3f} F1={mc['f1_macro']:.3f}  "
              f"CA: AUC={ma['auc_roc']:.3f} F1={ma['f1_macro']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "physics_baseline_results.csv", index=False)
    print("\n-- Mean across circuits --")
    print(df.groupby("model")[["auc_roc", "auc_pr", "f1_macro", "f1_xmode", "mcc"]]
          .mean().round(3).to_string())
    print("\nSaved physics_baseline_results.csv")


if __name__ == "__main__":
    main()
