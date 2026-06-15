from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from circuit_study_config import INPUTS_DIR, WORK_ROOT, ensure_work_dirs


MASS_KG = 798
DEFAULT_SPEED_THRESHOLD = 240.0


def parse_drivers(raw_value: str | float | int | None) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [item.strip().upper() for item in text.split(";") if item.strip()]


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
    if "drivers" not in manifest.columns:
        manifest["drivers"] = ""
    return manifest


def resolve_source_path(raw_dir: Path, source_file: str) -> Path:
    source_path = Path(str(source_file))
    if source_path.is_absolute():
        return source_path
    if source_path.parts and source_path.parts[0] == raw_dir.name:
        return raw_dir.parent / source_path
    return raw_dir / source_path


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        mapping = {"True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        mapped = series.map(mapping)
        if mapped.notna().any():
            return mapped.fillna(0).astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def engineer_row_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Speed", "RPM", "nGear", "Throttle", "Acceleration", "Engine_Load", "Elevation_Delta", "X", "Y", "Z", "Distance", "Tire_Age_Laps", "Time_Elapsed_Sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Brake" in df.columns:
        df["Brake"] = coerce_bool(df["Brake"])

    if "Compound" in df.columns:
        compound_order = sorted(df["Compound"].dropna().astype(str).unique())
        compound_map = {name: idx for idx, name in enumerate(compound_order)}
        df["Compound"] = df["Compound"].astype(str)
        df["Compound_Encoded"] = df["Compound"].map(compound_map).fillna(0).astype(int)
    else:
        df["Compound_Encoded"] = 0

    speed = df.get("Speed", pd.Series(0, index=df.index)).fillna(0)
    acceleration = df.get("Acceleration", pd.Series(0, index=df.index)).fillna(0)
    engine_load = df.get("Engine_Load", pd.Series(0, index=df.index)).fillna(0)

    if "Heavy_Braking" not in df.columns:
        df["Heavy_Braking"] = ((df.get("Brake", 0) == 1) | (acceleration < -3.0)).astype(int)

    if "High_Speed_Zone" not in df.columns:
        df["High_Speed_Zone"] = (speed > DEFAULT_SPEED_THRESHOLD).astype(int)

    if "Gear_Shift_Active" not in df.columns:
        if "nGear" in df.columns:
            df["Gear_Shift_Active"] = df["nGear"].diff().fillna(0).ne(0).astype(int)
        else:
            df["Gear_Shift_Active"] = 0

    df["Kinetic_Energy_MJ"] = 0.5 * MASS_KG * (speed / 3.6) ** 2 / 1e6
    df["Longitudinal_Force_N"] = MASS_KG * (acceleration / 3.6)
    df["Energy_Efficiency_Ratio"] = speed / (engine_load + 1.0)
    df["Speed_Rolling_Avg"] = speed.rolling(100, min_periods=1).mean()

    if "Optimal_Aero" not in df.columns:
        brake = coerce_bool(df["Brake"]) if "Brake" in df.columns else 0
        gear = pd.to_numeric(df.get("nGear", 0), errors="coerce").fillna(0)
        hsz = df["High_Speed_Zone"]
        heavy = df["Heavy_Braking"]
        elevation = df.get("Elevation_Delta", pd.Series(0, index=df.index)).fillna(0)
        df["Optimal_Aero"] = (
            (speed > DEFAULT_SPEED_THRESHOLD)
            & (brake == 0)
            & (gear >= 6)
            & (heavy == 0)
            & (hsz == 1)
            & (elevation > -3)
        ).astype(int)

    return df


def compute_time_elapsed_seconds(telemetry: pd.DataFrame) -> pd.Series:
    if "Time" in telemetry.columns:
        time_series = pd.to_timedelta(telemetry["Time"], errors="coerce")
        if time_series.notna().any():
            return time_series.dt.total_seconds().fillna(method="ffill").fillna(0.0)
    if "SessionTime" in telemetry.columns:
        time_series = pd.to_timedelta(telemetry["SessionTime"], errors="coerce")
        if time_series.notna().any():
            return time_series.dt.total_seconds().fillna(method="ffill").fillna(0.0)
    return pd.Series(np.arange(len(telemetry), dtype=float), index=telemetry.index)


def export_lap_telemetry(session, lap_row: pd.Series, circuit: str, event: str, season: int, session_type: str) -> pd.DataFrame:
    lap = lap_row
    telemetry = lap.get_telemetry()
    if telemetry.empty:
        return pd.DataFrame()

    telemetry = telemetry.copy()
    telemetry = telemetry.add_distance()
    telemetry["Circuit"] = circuit
    telemetry["Event"] = event
    telemetry["Season"] = int(season)
    telemetry["Session"] = session_type
    telemetry["Driver"] = str(lap["Driver"]).upper()
    telemetry["LapNumber"] = int(lap["LapNumber"])
    telemetry["Compound"] = str(lap.get("Compound", "")).upper()
    telemetry["Tire_Age_Laps"] = pd.to_numeric(lap.get("TyreLife", np.nan), errors="coerce")
    telemetry["Time_Elapsed_Sec"] = compute_time_elapsed_seconds(telemetry)
    telemetry["LapKey"] = (
        telemetry["Circuit"].astype(str)
        + "|"
        + telemetry["Event"].astype(str)
        + "|"
        + telemetry["Season"].astype(str)
        + "|"
        + telemetry["Driver"].astype(str)
        + "|L"
        + telemetry["LapNumber"].astype(str)
    )

    if "Acceleration" not in telemetry.columns:
        telemetry["Acceleration"] = telemetry["Speed"].diff().fillna(0)

    if "Engine_Load" not in telemetry.columns:
        telemetry["Engine_Load"] = telemetry.get("RPM", 0).fillna(0) * telemetry.get("Throttle", 0).fillna(0) / 100.0

    if "Elevation_Delta" not in telemetry.columns:
        telemetry["Elevation_Delta"] = telemetry.get("Z", 0).diff().fillna(0)

    telemetry = engineer_row_features(telemetry)
    return telemetry


def build_session(session_row: pd.Series, raw_dir: Path, out_dir: Path) -> Path | None:
    circuit = str(session_row["circuit"])
    event = str(session_row["event"])
    season = int(session_row["season"])
    session_type = str(session_row["session"])
    drivers = parse_drivers(session_row.get("drivers", ""))

    source_path = resolve_source_path(raw_dir, str(session_row["source_file"]))
    output_path = out_dir / source_path.name

    if output_path.exists() and source_path.exists() and output_path.samefile(source_path):
        return output_path

    print(f"Loading session: {season} {event} {session_type} ({circuit})")
    race = fastf1.get_session(season, event, session_type)
    race.load()

    laps = race.laps
    if laps.empty:
        warnings.warn(f"No laps found for {season} {event} {session_type}")
        return None

    if drivers:
        laps = laps.pick_drivers(drivers)

    if "IsAccurate" in laps.columns:
        accurate = laps["IsAccurate"].fillna(False).astype(bool)
        if accurate.any():
            laps = laps[accurate]

    if "LapTime" in laps.columns:
        laps = laps[laps["LapTime"].notna()]

    if "LapNumber" in laps.columns:
        laps = laps[pd.to_numeric(laps["LapNumber"], errors="coerce").fillna(0).astype(int) >= 2]

    lap_exports: list[pd.DataFrame] = []
    for _, lap_row in laps.iterrows():
        try:
            lap_df = export_lap_telemetry(race, lap_row, circuit, event, season, session_type)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Skipping lap {lap_row.get('LapNumber')} for {circuit} {lap_row.get('Driver')}: {exc}"
            )
            continue
        if not lap_df.empty:
            lap_exports.append(lap_df)

    if not lap_exports:
        warnings.warn(f"No telemetry exported for {circuit}")
        return None

    combined = pd.concat(lap_exports, ignore_index=True)
    combined = combined.sort_values(["Driver", "LapNumber", "Time_Elapsed_Sec"], kind="mergesort").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined):,} rows -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multi-circuit FastF1 telemetry into isolated CSVs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=INPUTS_DIR / "session_manifest.csv",
        help="Path to session_manifest.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=INPUTS_DIR / "raw",
        help="Directory to write per-circuit exports into",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=WORK_ROOT / ".fastf1_cache",
        help="FastF1 cache directory",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    ensure_work_dirs()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache_dir))

    manifest = read_manifest(args.manifest)
    outputs = []
    for row in manifest.itertuples(index=False):
        output = build_session(pd.Series(row._asdict()), args.raw_dir, args.raw_dir)
        if output is not None:
            outputs.append(output)

    print(f"Export complete. Files written: {len(outputs)}")


if __name__ == "__main__":
    main()