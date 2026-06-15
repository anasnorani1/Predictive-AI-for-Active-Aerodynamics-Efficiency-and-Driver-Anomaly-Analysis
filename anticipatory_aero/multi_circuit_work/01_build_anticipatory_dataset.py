"""
01_build_anticipatory_dataset.py
=================================
Builds the leakage-free anticipatory aerodynamic mode prediction dataset.

Pipeline:
  1. Optionally pull missing circuit CSVs from FastF1 (--pull-missing).
  2. Load each circuit CSV; keep only REAL FastF1 channels and physics-derived features.
     Engine_Load and Energy_Efficiency_Ratio are intentionally excluded — they are
     fabricated proxies (RPM*Throttle/100), not direct sensor readings.
  3. Build sliding windows of length W; label each window at horizon H (strictly
     outside the input window, preventing any future-state leakage).
  4. Construct leave-one-circuit-out (LOCO) splits; fit a StandardScaler on each
     fold's train windows only.
  5. Save one .npz per horizon H (raw unscaled windows) and one .pkl scaler per fold.
  6. Print a stats summary — paste the output back so we can verify the dataset.

Usage:
  cd multi_circuit_work
  python 01_build_anticipatory_dataset.py              # uses existing CSVs only
  python 01_build_anticipatory_dataset.py --pull-missing  # fetch from FastF1 first
  python 01_build_anticipatory_dataset.py --w 50 --horizons 1 5 10 25 50

Outputs (all under multi_circuit_work/processed/):
  windows_H{H:03d}.npz        — X (N,W,F) float32, y (N,) int8, circuits, lap_keys
  scaler_holdout_{C}_H{H:03d}.pkl  — StandardScaler fit on train windows for each fold
  dataset_stats.txt           — human-readable summary; paste this back
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
WORK_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = WORK_ROOT / "inputs"
RAW_DIR = INPUTS_DIR / "raw"
PROCESSED_DIR = WORK_ROOT / "processed"

MASS_KG = 798.0          # F1 car minimum weight (kg), 2026 regulations
SPEED_THRESHOLD = 240.0  # km/h — X-Mode activation threshold
GEAR_THRESHOLD = 6       # minimum gear for X-Mode

# ── feature columns ───────────────────────────────────────────────────────────
# Only real FastF1 channels and quantities derived exclusively from them.
# Excluded: Engine_Load (RPM*Throttle/100 — fabricated), Energy_Efficiency_Ratio
#           (uses Engine_Load), High_Speed_Zone (threshold on Speed — leaky input),
#           Active_Aero_State / Optimal_Aero (the instantaneous label itself).
FEATURE_COLS = [
    "Speed",               # km/h — direct sensor
    "RPM",                 # direct sensor
    "nGear",               # direct sensor
    "Throttle",            # % — direct sensor
    "Brake",               # binary — direct sensor
    "X",                   # m — GPS coordinate
    "Y",                   # m
    "Z",                   # m
    "Acceleration",        # km/h per sample — derived from Speed via diff()
    "Elevation_Delta",     # m per sample — derived from Z via diff()
    "Kinetic_Energy_MJ",   # 0.5 * m * (v/3.6)^2 / 1e6 — derived from Speed
    "Longitudinal_Force_N", # m * (a/3.6) — derived from Acceleration
]
F = len(FEATURE_COLS)


# ── data loading ──────────────────────────────────────────────────────────────

def _coerce_brake(series: pd.Series) -> pd.Series:
    """Convert Brake column (bool, 'True'/'False' strings, 0/1) to float."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.float32)
    if series.dtype == object:
        return series.map({"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0,
                           "1": 1.0, "0": 0.0}).fillna(0.0).astype(np.float32)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32)


def load_and_clean(csv_path: Path, circuit: str) -> pd.DataFrame:
    """
    Load a per-circuit CSV and return a clean DataFrame with FEATURE_COLS plus
    the columns needed for grouping (Circuit, Driver, LapNumber, LapKey,
    Time_Elapsed_Sec).  NaN rows in any feature column are dropped.
    """
    df = pd.read_csv(csv_path)
    df["Circuit"] = circuit

    # coerce numeric columns
    for col in ["Speed", "RPM", "nGear", "Throttle", "X", "Y", "Z",
                "LapNumber", "Time_Elapsed_Sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Brake" in df.columns:
        df["Brake"] = _coerce_brake(df["Brake"])
    else:
        df["Brake"] = 0.0

    # sort within laps before computing diffs
    df = df.sort_values(["Driver", "LapNumber", "Time_Elapsed_Sec"],
                        kind="mergesort").reset_index(drop=True)

    # Acceleration — diff of Speed within each lap
    if "Acceleration" not in df.columns:
        df["Acceleration"] = (
            df.groupby(["Driver", "LapNumber"])["Speed"]
            .diff()
            .fillna(0.0)
        )
    else:
        df["Acceleration"] = pd.to_numeric(df["Acceleration"], errors="coerce").fillna(0.0)

    # Elevation_Delta — diff of Z within each lap
    if "Elevation_Delta" not in df.columns:
        df["Elevation_Delta"] = (
            df.groupby(["Driver", "LapNumber"])["Z"]
            .diff()
            .fillna(0.0)
        )
    else:
        df["Elevation_Delta"] = pd.to_numeric(df["Elevation_Delta"], errors="coerce").fillna(0.0)

    # physics-derived features
    speed = df["Speed"].fillna(0.0)
    accel = df["Acceleration"].fillna(0.0)
    df["Kinetic_Energy_MJ"] = 0.5 * MASS_KG * (speed / 3.6) ** 2 / 1e6
    df["Longitudinal_Force_N"] = MASS_KG * (accel / 3.6)

    # LapKey — unique identifier for one driver's one lap at this circuit
    df["LapKey"] = (
        circuit
        + "|"
        + df["Driver"].astype(str)
        + "|L"
        + df["LapNumber"].fillna(-1).astype(int).astype(str)
    )

    # drop rows with NaN in any feature column
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        warnings.warn(f"{circuit}: dropped {dropped} rows with NaN in features.")

    return df


# ── windowing ─────────────────────────────────────────────────────────────────

def build_windows(
    dfs: list[pd.DataFrame],
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build all valid sliding windows across all circuits and laps.

    For window start index i within a lap of length L:
      Input  : samples [i, i+1, ..., i+W-1]            ← past W steps
      Label  : sample  [i + W - 1 + H] = [i + W + H - 1]  ← H steps ahead of last input

    The label index is strictly greater than any input index, so no future state
    leaks into the input. Windows that would cross a lap boundary are never formed
    (we window within each LapKey group separately).

    Minimum lap length for ≥1 window: W + H samples.

    Returns
    -------
    X        : float32 (N, W, F)
    y        : int8    (N,)        — 1 if Speed[label] >= 240 AND nGear[label] >= 6
    circuits : object  (N,)        — circuit name per window
    lap_keys : object  (N,)        — LapKey per window
    """
    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_circuits: list[str] = []
    all_lapkeys: list[str] = []

    for df in dfs:
        circuit = df["Circuit"].iloc[0]

        for lap_key, lap_df in df.groupby("LapKey", sort=False):
            lap_df = lap_df.sort_values("Time_Elapsed_Sec").reset_index(drop=True)
            L = len(lap_df)

            if L < W + H:
                continue

            feat = lap_df[FEATURE_COLS].to_numpy(dtype=np.float32)  # (L, F)
            speed_arr = lap_df["Speed"].to_numpy(dtype=np.float32)
            gear_arr = lap_df["nGear"].to_numpy(dtype=np.float32)

            n_windows = L - W - H + 1  # number of valid starting positions
            for i in range(n_windows):
                label_idx = i + W + H - 1  # strictly > last input index (i+W-1)
                y_val = int(
                    speed_arr[label_idx] >= SPEED_THRESHOLD
                    and gear_arr[label_idx] >= GEAR_THRESHOLD
                )
                all_X.append(feat[i : i + W])
                all_y.append(y_val)
                all_circuits.append(circuit)
                all_lapkeys.append(str(lap_key))

    if not all_X:
        raise ValueError(
            f"No windows produced for W={W}, H={H}. "
            "Check that loaded laps are long enough (need ≥ W+H samples per lap)."
        )

    return (
        np.stack(all_X, axis=0),             # (N, W, F)
        np.array(all_y, dtype=np.int8),       # (N,)
        np.array(all_circuits, dtype=object), # (N,)
        np.array(all_lapkeys, dtype=object),  # (N,)
    )


# ── LOCO scalers ──────────────────────────────────────────────────────────────

def fit_loco_scalers(
    X: np.ndarray,
    circuits: np.ndarray,
    circuit_list: list[str],
) -> dict[str, StandardScaler]:
    """
    For each held-out circuit, fit StandardScaler on the train windows only.
    The scaler is fit on flattened (N_train, W*F) data so all time steps and
    features are normalised consistently.
    """
    N, W, Fv = X.shape
    scalers: dict[str, StandardScaler] = {}
    for held_out in circuit_list:
        train_mask = circuits != held_out
        X_train_flat = X[train_mask].reshape(-1, W * Fv)
        scaler = StandardScaler()
        scaler.fit(X_train_flat)
        scalers[held_out] = scaler
    return scalers


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pull-missing", action="store_true",
                   help="Call export_fastf1_multi_circuit.py to fetch missing circuit CSVs first.")
    p.add_argument("--w", type=int, default=50,
                   help="Window length in samples (default: 50 ≈ 5 s at 10 Hz).")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50],
                   help="Anticipation horizons in samples (default: 1 5 10 25 50).")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test with whatever circuits are available (skips LOCO ≥2 requirement).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── manifest ───────────────────────────────────────────────────────────────
    manifest_path = INPUTS_DIR / "session_manifest.csv"
    if not manifest_path.exists():
        manifest_path = INPUTS_DIR / "session_manifest.example.csv"
    manifest = pd.read_csv(manifest_path)

    # ── optional: pull missing CSVs ───────────────────────────────────────────
    missing = [r.source_file for r in manifest.itertuples(index=False)
               if not (RAW_DIR / r.source_file).exists()]
    if missing and args.pull_missing:
        print(f"Pulling {len(missing)} missing circuit(s) from FastF1: {missing}")
        subprocess.run(
            [sys.executable, str(WORK_ROOT / "export_fastf1_multi_circuit.py")],
            check=True,
        )
    elif missing:
        print(f"WARNING: {len(missing)} circuit CSV(s) not found: {missing}")
        print("  Run with --pull-missing to fetch, or copy CSVs manually to:")
        print(f"  {RAW_DIR}")

    # ── load available CSVs ───────────────────────────────────────────────────
    dfs: list[pd.DataFrame] = []
    circuit_list: list[str] = []

    for row in manifest.itertuples(index=False):
        raw_path = RAW_DIR / row.source_file
        if not raw_path.exists():
            print(f"  Skipping {row.circuit} — CSV not found.")
            continue
        print(f"Loading {row.circuit} ({row.source_file}) ...", end=" ", flush=True)
        df = load_and_clean(raw_path, row.circuit)
        xmode_instant = ((df["Speed"] >= SPEED_THRESHOLD) & (df["nGear"] >= GEAR_THRESHOLD)).mean()
        print(f"{len(df):,} rows | {df['LapKey'].nunique()} laps | "
              f"drivers={df['Driver'].nunique()} | "
              f"instantaneous X-mode={xmode_instant:.3f}")
        dfs.append(df)
        circuit_list.append(row.circuit)

    n_circuits = len(circuit_list)
    if n_circuits < 2 and not args.test:
        raise RuntimeError(
            f"Only {n_circuits} circuit(s) loaded — need ≥ 2 for LOCO evaluation.\n"
            "Run with --pull-missing or place CSVs manually.\n"
            "Use --test to smoke-test windowing with however many circuits you have."
        )

    W = args.w
    horizons = args.horizons

    stats_lines: list[str] = [
        "=" * 72,
        "ANTICIPATORY AERO DATASET STATISTICS",
        f"Window W={W} samples (~{W/10:.1f} s at 10 Hz)",
        f"Horizons H={horizons} samples (~{[h/10 for h in horizons]} s)",
        f"Features ({F}): {', '.join(FEATURE_COLS)}",
        f"Label: y=1 iff Speed[t+H] >= {SPEED_THRESHOLD} AND nGear[t+H] >= {GEAR_THRESHOLD}",
        f"Circuits ({n_circuits}): {circuit_list}",
        "=" * 72,
        "",
        "── Per-circuit telemetry ──────────────────────────────────────────────",
    ]

    for df, circuit in zip(dfs, circuit_list):
        xm = ((df["Speed"] >= SPEED_THRESHOLD) & (df["nGear"] >= GEAR_THRESHOLD)).mean()
        stats_lines.append(
            f"  {circuit:12s}  rows={len(df):7,}  laps={df['LapKey'].nunique():3d}  "
            f"drivers={df['Driver'].nunique()}  inst_X-mode={xm:.3f}"
        )

    stats_lines += ["", "── Per-horizon windowed dataset ───────────────────────────────────────"]

    for H in horizons:
        print(f"\nBuilding windows  W={W}, H={H} ...", end=" ", flush=True)
        X, y, circuits, lap_keys = build_windows(dfs, W, H)
        N = len(y)
        xmode_rate = float(y.mean())
        print(f"N={N:,}  X-mode={xmode_rate:.3f}  shape={X.shape}")

        # save raw (unscaled) windows — scalers are fold-specific, applied at train time
        npz_path = PROCESSED_DIR / f"windows_H{H:03d}.npz"
        np.savez_compressed(npz_path, X=X, y=y, circuits=circuits, lap_keys=lap_keys,
                            feature_cols=np.array(FEATURE_COLS))
        print(f"  Saved: {npz_path.name}  ({npz_path.stat().st_size/1e6:.1f} MB)")

        # fit and save per-fold scalers (train-only); skip empty folds in --test mode
        if n_circuits >= 2:
            scalers = fit_loco_scalers(X, circuits, circuit_list)
            for held_out, scaler in scalers.items():
                pkl_path = PROCESSED_DIR / f"scaler_holdout_{held_out}_H{H:03d}.pkl"
                with open(pkl_path, "wb") as fh:
                    pickle.dump(scaler, fh, protocol=4)
        else:
            print("  (Skipping LOCO scalers — only 1 circuit loaded)")

        # stats block
        stats_lines.append(f"\n  H={H:3d} (~{H/10:.1f}s ahead)  "
                           f"total_windows={N:8,}  overall_X-mode={xmode_rate:.3f}")
        for held_out in circuit_list:
            test_mask = circuits == held_out
            train_mask = ~test_mask
            n_tr = int(train_mask.sum())
            n_te = int(test_mask.sum())
            r_tr = float(y[train_mask].mean()) if n_tr else float("nan")
            r_te = float(y[test_mask].mean()) if n_te else float("nan")
            stats_lines.append(
                f"    fold hold-out={held_out:12s}  "
                f"train={n_tr:7,} (X-mode={r_tr:.3f})  "
                f"test={n_te:7,} (X-mode={r_te:.3f})"
            )

    stats_lines += [
        "",
        "── Class-imbalance note ───────────────────────────────────────────────",
        "  Focal loss (gamma=2, alpha=0.25) is used during training to address imbalance.",
        "  Report both F1-macro and AUC-PR alongside AUC-ROC.",
        "",
        "── Leakage check ──────────────────────────────────────────────────────",
        f"  Input window indices:  [i, i+1, ..., i+W-1]",
        f"  Label index:           i + W + H - 1   (H={min(horizons)}..{max(horizons)} steps ahead)",
        "  Label index > last input index: ALWAYS TRUE by construction.",
        "  No future state leaks into the input window.",
    ]

    stats_text = "\n".join(stats_lines)
    stats_path = PROCESSED_DIR / "dataset_stats.txt"
    stats_path.write_text(stats_text, encoding="utf-8")

    print("\n" + stats_text)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
