from pathlib import Path


WORK_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = WORK_ROOT / "inputs"
PROCESSED_DIR = WORK_ROOT / "processed"
MODELS_DIR = WORK_ROOT / "models"
GRAPHS_DIR = WORK_ROOT / "graphs"
NOTEBOOKS_DIR = WORK_ROOT / "notebooks"

TARGET_CIRCUITS = [
    "Monza",
    "Monaco",
    "Silverstone",
    "Spa",
    "Suzuka",
]

TARGET_DRIVERS = ["VER", "HAD"]

HORIZONS = [1, 5, 10, 15, 20, 30]
WINDOW_LENGTHS = [5, 10, 20, 40]

SPLIT_POLICIES = {
    "within_circuit": "chronological",
    "loco": "leave_one_circuit_out",
}


def ensure_work_dirs() -> None:
    for directory in [INPUTS_DIR, PROCESSED_DIR, MODELS_DIR, GRAPHS_DIR, NOTEBOOKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_work_dirs()
    print(f"Working area initialized at: {WORK_ROOT}")