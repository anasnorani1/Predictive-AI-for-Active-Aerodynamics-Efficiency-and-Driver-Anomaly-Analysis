# Input Staging

Place per-circuit FastF1 exports here before running the isolated multi-circuit build script.

Expected layout:

- `raw/` contains one CSV per circuit/session export
- `session_manifest.csv` describes each file and the circuit label to use

Recommended source naming:

- `raw/monza_red_bull.csv`
- `raw/monaco_red_bull.csv`
- `raw/silverstone_red_bull.csv`
- `raw/suzuka_red_bull.csv`

In `session_manifest.csv`, use the file name relative to `raw/`, for example `monza_red_bull.csv`.

The build script will keep all outputs under `multi_circuit_work/processed`, `multi_circuit_work/graphs`, and `multi_circuit_work/models`.