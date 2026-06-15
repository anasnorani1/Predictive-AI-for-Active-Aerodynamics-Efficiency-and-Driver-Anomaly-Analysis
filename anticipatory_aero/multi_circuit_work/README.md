# Multi-Circuit Working Area

This folder is reserved for the anticipatory aero rework so it does not mix with the existing Suzuka-only pipeline outputs.

## Intended layout

- `inputs/` for raw FastF1 exports and any per-circuit source files
- `processed/` for combined multi-circuit tables and split artefacts
- `models/` for LOCO and within-circuit model outputs
- `graphs/` for horizon plots, cross-circuit figures, and ablations
- `notebooks/` for any new exploratory notebooks tied to the rework

## Included circuit set

- Monza
- Monaco
- Silverstone or Spa
- Suzuka

## Split policy

- Within-circuit chronological split
- Leave-one-circuit-out evaluation
- No leakage across lap or window boundaries

## Working rule

Keep every new file for the multi-circuit study inside this folder tree so the original `artefacts/`, `graphs/`, and `models/` folders remain untouched.

## Quick Start

1. Drop per-circuit CSV exports into `multi_circuit_work/inputs/raw/`.
2. Copy `multi_circuit_work/inputs/session_manifest.example.csv` to `multi_circuit_work/inputs/session_manifest.csv` and adjust the source file names if needed.
3. Run `python multi_circuit_work/export_fastf1_multi_circuit.py` from the repository root to generate the raw CSVs.
4. Run `python multi_circuit_work/build_multi_circuit_study.py` from the repository root to build the combined dataset and split manifests.

The script writes the combined dataset, lap manifests, circuit summary, and LOCO split tables to `multi_circuit_work/processed/`.