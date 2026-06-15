"""
extract_xai_package.py
======================
Run this once after downloading xai_package.zip from Kaggle Output tab.

Usage:
    cd multi_circuit_work
    python extract_xai_package.py --zip path/to/xai_package.zip

Places files exactly where 05_xai.py expects them:
    models/   -> Transformer_fold_Monaco_H010.pt, GRU_fold_Monza_H010.pt, …
    processed/ -> windows_H010.npz, scaler_holdout_Monaco_H010.pkl, …
"""

import argparse
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--zip", required=True,
                   help="Path to xai_package.zip downloaded from Kaggle.")
    args = p.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    dest = Path(__file__).resolve().parent  # multi_circuit_work/
    print(f"Extracting {zip_path.name}  ->  {dest}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pt_files  = [n for n in names if n.endswith(".pt")]
        npz_files = [n for n in names if n.endswith(".npz")]
        pkl_files = [n for n in names if n.endswith(".pkl")]

        print(f"  Contains: {len(pt_files)} models, "
              f"{len(npz_files)} window files, {len(pkl_files)} scalers")

        for name in names:
            target = dest / name          # e.g. models/CNN_fold_Monaco_H010.pt
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))

    print("\nExtracted:")
    for f in sorted((dest / "models").glob("*.pt")):
        print(f"  models/{f.name}")
    for f in sorted((dest / "processed").glob("windows*.npz")):
        print(f"  processed/{f.name}")
    for f in sorted((dest / "processed").glob("scaler*.pkl")):
        print(f"  processed/{f.name}")

    print("\nReady. Run XAI analysis:")
    print("  python 05_xai.py --H 10 --held-out Monaco --model Transformer")
    print("  python 05_xai.py --H 10 --held-out Monaco --model GRU")
    print("  python 05_xai.py --H 10 --held-out Monza  --model Transformer")
    print("  python 05_xai.py --H 10 --held-out Monza  --model GRU")


if __name__ == "__main__":
    main()
