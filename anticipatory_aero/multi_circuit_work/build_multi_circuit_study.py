from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from circuit_study_config import GRAPHS_DIR, INPUTS_DIR, MODELS_DIR, PROCESSED_DIR, WORK_ROOT, ensure_work_dirs


MASS_KG = 798
ROLLING_WINDOW = 100
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SPEED_THRESHOLD = 240.0


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        fallback = manifest_path.with_name("session_manifest.example.csv")
        if not fallback.exists():
            raise FileNotFoundError(
                f"No manifest found at {manifest_path} and no example manifest at {fallback}."
            )
        manifest_path = fallback

    manifest = pd.read_csv(manifest_path)
    required = {"circuit", "event", "season", "session", "source_file"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return manifest


def load_source_csv(raw_path: Path, circuit: str, event: str, season: int, session_type: str) -> pd.DataFrame:
    if not raw_path.exists():
        warnings.warn(f"Missing source export: {raw_path}")
        return pd.DataFrame()

    df = pd.read_csv(raw_path)
    df["Circuit"] = circuit
    df["Event"] = event
    df["Season"] = int(season)
    df["Session"] = session_type
    return df


def resolve_source_path(raw_dir: Path, source_file: str) -> Path:
    source_path = Path(str(source_file))
    if source_path.is_absolute():
        return source_path
    if source_path.parts and source_path.parts[0] == raw_dir.name:
        return raw_dir.parent / source_path
    return raw_dir / source_path


def coerce_binary(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        mapping = {"True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        if series.dropna().isin(mapping).all():
            return series.map(mapping).astype("Int64").fillna(0).astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "LapNumber" in df.columns:
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")

    if "Brake" in df.columns:
        df["Brake"] = coerce_binary(df["Brake"])

    for col in ["Heavy_Braking", "High_Speed_Zone", "Gear_Shift_Active", "Optimal_Aero"]:
        if col in df.columns:
            df[col] = coerce_binary(df[col])

    if "Compound" in df.columns:
        compound_order = sorted(df["Compound"].dropna().astype(str).unique())
        compound_map = {name: idx for idx, name in enumerate(compound_order)}
        df["Compound"] = df["Compound"].astype(str)
        df["Compound_Encoded"] = df["Compound"].map(compound_map).fillna(0).astype(int)
    else:
        df["Compound_Encoded"] = 0

    if "Engine_Load" not in df.columns:
        if {"RPM", "Throttle"}.issubset(df.columns):
            df["Engine_Load"] = pd.to_numeric(df["RPM"], errors="coerce").fillna(0) * pd.to_numeric(df["Throttle"], errors="coerce").fillna(0) / 100.0
        else:
            df["Engine_Load"] = 0.0

    if "Acceleration" not in df.columns:
        if {"Speed", "Time_Elapsed_Sec"}.issubset(df.columns):
            ordered = df.sort_values(["Circuit", "Event", "Season", "Driver", "LapNumber", "Time_Elapsed_Sec"], kind="mergesort")
            accel = (
                ordered.groupby(["Circuit", "Event", "Season", "Driver", "LapNumber"], sort=False)["Speed"]
                .diff()
                .fillna(0.0)
            )
            ordered["Acceleration"] = accel
            df = ordered.sort_index()
        else:
            df["Acceleration"] = 0.0

    if "Elevation_Delta" not in df.columns:
        if "Z" in df.columns:
            ordered = df.sort_values(["Circuit", "Event", "Season", "Driver", "LapNumber", "Time_Elapsed_Sec"], kind="mergesort")
            ordered["Elevation_Delta"] = (
                ordered.groupby(["Circuit", "Event", "Season", "Driver", "LapNumber"], sort=False)["Z"]
                .diff()
                .fillna(0.0)
            )
            df = ordered.sort_index()
        else:
            df["Elevation_Delta"] = 0.0

    if "High_Speed_Zone" not in df.columns and "Speed" in df.columns:
        df["High_Speed_Zone"] = (pd.to_numeric(df["Speed"], errors="coerce").fillna(0) > DEFAULT_SPEED_THRESHOLD).astype(int)

    if "Heavy_Braking" not in df.columns:
        accel = pd.to_numeric(df.get("Acceleration", 0), errors="coerce").fillna(0)
        brake = coerce_binary(df["Brake"]) if "Brake" in df.columns else pd.Series(0, index=df.index)
        df["Heavy_Braking"] = ((brake == 1) | (accel < -3.0)).astype(int)

    if "Gear_Shift_Active" not in df.columns:
        if {"Driver", "LapNumber", "Time_Elapsed_Sec", "nGear"}.issubset(df.columns):
            ordered = df.sort_values(["Circuit", "Event", "Season", "Driver", "LapNumber", "Time_Elapsed_Sec"], kind="mergesort")
            gear_shift = (
                ordered.groupby(["Circuit", "Event", "Season", "Driver", "LapNumber"], sort=False)["nGear"]
                .diff()
                .fillna(0)
                .ne(0)
                .astype(int)
            )
            ordered["Gear_Shift_Active"] = gear_shift
            df = ordered.sort_index()
        else:
            df["Gear_Shift_Active"] = 0

    if "Speed_Rolling_Avg" not in df.columns:
        if {"Driver", "LapNumber", "Speed"}.issubset(df.columns):
            ordered = df.sort_values(["Circuit", "Event", "Season", "Driver", "LapNumber", "Time_Elapsed_Sec"], kind="mergesort")
            ordered["Speed_Rolling_Avg"] = (
                ordered.groupby(["Circuit", "Event", "Season", "Driver", "LapNumber"], sort=False)["Speed"]
                .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
            )
            df = ordered.sort_index()
        else:
            df["Speed_Rolling_Avg"] = pd.to_numeric(df.get("Speed", 0), errors="coerce").fillna(0)

    speed = pd.to_numeric(df.get("Speed", 0), errors="coerce").fillna(0)
    acceleration = pd.to_numeric(df.get("Acceleration", 0), errors="coerce").fillna(0)
    engine_load = pd.to_numeric(df.get("Engine_Load", 0), errors="coerce").fillna(0)
    throttle = pd.to_numeric(df.get("Throttle", 0), errors="coerce").fillna(0)

    df["Kinetic_Energy_MJ"] = 0.5 * MASS_KG * (speed / 3.6) ** 2 / 1e6
    df["Longitudinal_Force_N"] = MASS_KG * (acceleration / 3.6)
    df["Energy_Efficiency_Ratio"] = speed / (engine_load + 1.0)

    if "Optimal_Aero" not in df.columns:
        brake = coerce_binary(df["Brake"]) if "Brake" in df.columns else 0
        gear = pd.to_numeric(df.get("nGear", 0), errors="coerce").fillna(0)
        hsz = coerce_binary(df["High_Speed_Zone"]) if "High_Speed_Zone" in df.columns else (speed > DEFAULT_SPEED_THRESHOLD).astype(int)
        heavy = coerce_binary(df["Heavy_Braking"]) if "Heavy_Braking" in df.columns else (acceleration < -3.0).astype(int)
        elevation = pd.to_numeric(df.get("Elevation_Delta", 0), errors="coerce").fillna(0)

        df["Optimal_Aero"] = (
            (speed > DEFAULT_SPEED_THRESHOLD)
            & (brake == 0)
            & (gear >= 6)
            & (heavy == 0)
            & (hsz == 1)
            & (elevation > -3)
        ).astype(int)

    return df


def add_lap_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Driver"] = df["Driver"].astype(str)
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(-1).astype(int)
    df["Time_Elapsed_Sec"] = pd.to_numeric(df.get("Time_Elapsed_Sec", 0), errors="coerce").fillna(0.0)
    df["LapKey"] = (
        df["Circuit"].astype(str)
        + "|"
        + df["Event"].astype(str)
        + "|"
        + df["Season"].astype(str)
        + "|"
        + df["Driver"].astype(str)
        + "|L"
        + df["LapNumber"].astype(str)
    )
    return df


def build_lap_manifest(df: pd.DataFrame) -> pd.DataFrame:
    lap_manifest = (
        df.groupby("LapKey", as_index=False)
        .agg(
            Circuit=("Circuit", "first"),
            Event=("Event", "first"),
            Season=("Season", "first"),
            Driver=("Driver", "first"),
            LapNumber=("LapNumber", "first"),
            StartTime=("Time_Elapsed_Sec", "min"),
            EndTime=("Time_Elapsed_Sec", "max"),
            Samples=("LapKey", "size"),
        )
        .sort_values(["Circuit", "StartTime", "Driver", "LapNumber"], kind="mergesort")
        .reset_index(drop=True)
    )

    lap_manifest["WithinCircuitOrder"] = lap_manifest.groupby("Circuit").cumcount() + 1
    lap_manifest["WithinCircuitLapCount"] = lap_manifest.groupby("Circuit")["LapKey"].transform("size")
    lap_manifest["WithinCircuitTrainCutoff"] = np.floor(lap_manifest["WithinCircuitLapCount"] * DEFAULT_TRAIN_FRACTION).clip(lower=1).astype(int)
    lap_manifest["WithinCircuitSplit"] = np.where(
        lap_manifest["WithinCircuitOrder"] <= lap_manifest["WithinCircuitTrainCutoff"],
        "train",
        "test",
    )
    return lap_manifest


def build_loco_manifest(lap_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    circuits = lap_manifest["Circuit"].dropna().unique().tolist()
    for fold_id, held_out in enumerate(circuits, start=1):
        fold = lap_manifest[["LapKey", "Circuit"]].copy()
        fold["FoldId"] = fold_id
        fold["HeldOutCircuit"] = held_out
        fold["Partition"] = np.where(fold["Circuit"] == held_out, "test", "train")
        rows.append(fold)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["LapKey", "Circuit", "FoldId", "HeldOutCircuit", "Partition"])


def summarize_by_circuit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Circuit", as_index=False)
        .agg(
            Samples=("Circuit", "size"),
            Drivers=("Driver", "nunique"),
            Laps=("LapKey", "nunique"),
            X_Mode_Rate=("Optimal_Aero", "mean"),
            Mean_Speed=("Speed", "mean"),
            Mean_Throttle=("Throttle", "mean"),
        )
        .sort_values("Circuit")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated multi-circuit aero study artefacts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=INPUTS_DIR / "session_manifest.csv",
        help="Path to the session manifest CSV. Falls back to session_manifest.example.csv if missing.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=INPUTS_DIR / "raw",
        help="Directory containing the per-circuit CSV exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_work_dirs()

    manifest = read_manifest(args.manifest)
    sources = []

    for row in manifest.itertuples(index=False):
        raw_path = resolve_source_path(args.raw_dir, str(row.source_file))
        df_source = load_source_csv(raw_path, row.circuit, row.event, row.season, row.session)
        if df_source.empty:
            continue
        sources.append(df_source)

    if not sources:
        raise RuntimeError(
            f"No source CSVs were loaded. Put per-circuit exports under {args.raw_dir} and update the manifest."
        )

    combined = pd.concat(sources, ignore_index=True)
    combined = add_engineered_features(combined)
    combined = add_lap_keys(combined)
    combined = combined.sort_values(["Circuit", "Driver", "LapNumber", "Time_Elapsed_Sec"], kind="mergesort").reset_index(drop=True)

    processed_path = PROCESSED_DIR / "multi_circuit_preprocessed.csv"
    combined.to_csv(processed_path, index=False)

    lap_manifest = build_lap_manifest(combined)
    lap_manifest.to_csv(PROCESSED_DIR / "lap_manifest.csv", index=False)

    loco_manifest = build_loco_manifest(lap_manifest)
    loco_manifest.to_csv(PROCESSED_DIR / "loco_fold_lap_manifest.csv", index=False)

    summary = summarize_by_circuit(combined)
    summary.to_csv(PROCESSED_DIR / "circuit_summary.csv", index=False)

    within_split = lap_manifest[["LapKey", "Circuit", "Driver", "LapNumber", "WithinCircuitSplit", "Samples"]].copy()
    within_split.to_csv(PROCESSED_DIR / "within_circuit_lap_split.csv", index=False)

    combined.to_csv(PROCESSED_DIR / "multi_circuit_preprocessed_with_splits.csv", index=False)

    print(f"Saved combined dataset: {processed_path}")
    print(f"Saved lap manifest: {PROCESSED_DIR / 'lap_manifest.csv'}")
    print(f"Saved LOCO fold manifest: {PROCESSED_DIR / 'loco_fold_lap_manifest.csv'}")
    print(f"Saved circuit summary: {PROCESSED_DIR / 'circuit_summary.csv'}")
    print(f"Rows: {len(combined):,} | Circuits: {combined['Circuit'].nunique()} | Laps: {combined['LapKey'].nunique()}")


if __name__ == "__main__":
    main()