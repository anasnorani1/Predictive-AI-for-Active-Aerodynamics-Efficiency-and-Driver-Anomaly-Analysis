"""
03_deep_models.py
==================
Temporal deep learning models for anticipatory aero mode prediction.

Architectures:
  CNN       : 1D causal convolutional network
  LSTM      : Causal LSTM (bidirectional=False)
  GRU       : Causal GRU
  TCN       : Temporal Convolutional Network (Bai et al. 2018, dilated causal)
  Transformer: Causal Transformer encoder (masked self-attention)

All models trained with focal loss + AdamW + early stopping under LOCO splits.
Scalers fitted on train-only (loaded from 01_build_anticipatory_dataset.py outputs).

Usage:
  cd multi_circuit_work
  python 03_deep_models.py                        # H=10, all models
  python 03_deep_models.py --horizons 1 5 10 25 50 --models CNN LSTM TCN Transformer
  python 03_deep_models.py --horizons 10 --fast   # fewer epochs for smoke-test

Outputs (all under multi_circuit_work/):
  processed/deep_results.csv
  processed/deep_preds/{model}_{held_out}_H{H:03d}.npz  (y_prob, y_true, y_pred)
  models/{model}_fold_{held_out}_H{H:03d}.pt             (best checkpoint per fold)
"""
from __future__ import annotations

import argparse
import copy
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

WORK_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = WORK_ROOT / "processed"
PREDS_DIR = PROCESSED_DIR / "deep_preds"
MODELS_DIR = WORK_ROOT / "models"

SEED = 42


# ── reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ── focal loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al. 2017).
    alpha: weight for the positive class (X-mode).
    gamma: focusing parameter; gamma=0 recovers weighted BCE.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: (B,) — raw logits, binary {0, 1}
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t = torch.exp(-bce)                   # p if y=1, (1-p) if y=0
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ── model architectures ────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Conv1d with left-only padding — strictly causal."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self._pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, T)
        return self.conv(F.pad(x, (self._pad, 0)))


class CNN1D(nn.Module):
    """
    Three-layer 1D causal CNN with global average pooling.
    Input: (B, W, F) — transposed internally to (B, F, W).
    """
    def __init__(self, W: int, F: int, channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(F, channels, 5), nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(channels, channels * 2, 5), nn.BatchNorm1d(channels * 2), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(channels * 2, channels * 2, 5), nn.BatchNorm1d(channels * 2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.ReLU(), nn.Dropout(dropout + 0.1),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F) → (B,)
        x = self.net(x.transpose(1, 2))       # (B, F, W) → (B, C, W)
        x = x.mean(dim=-1)                    # global average pool → (B, C)
        return self.head(x).squeeze(-1)


class CausalLSTM(nn.Module):
    """Two-layer causal LSTM, last hidden state → classifier."""
    def __init__(self, F: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(F, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout + 0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F) → (B,)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CausalGRU(nn.Module):
    """Two-layer causal GRU, last hidden state → classifier."""
    def __init__(self, F: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(F, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout + 0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F) → (B,)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.proj is None else self.proj(x)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        return F.relu(x + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (Bai et al. 2018).
    4 residual blocks with dilations 1,2,4,8 — receptive field ≥ 121 > W=50.
    Input: (B, W, F) — transposed internally.
    """
    def __init__(self, F: int, channels: int = 64, kernel: int = 5, dropout: float = 0.2):
        super().__init__()
        dilations = [1, 2, 4, 8]
        layers: list[nn.Module] = []
        in_ch = F
        for d in dilations:
            layers.append(_TCNBlock(in_ch, channels, kernel, d, dropout))
            in_ch = channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.ReLU(), nn.Dropout(dropout + 0.1),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F) → (B,)
        x = self.network(x.transpose(1, 2))   # (B, F, W) → (B, channels, W)
        return self.head(x[:, :, -1]).squeeze(-1)  # last time step


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CausalTransformer(nn.Module):
    """
    Transformer encoder with causal (upper-triangular) self-attention mask.
    Processes (B, W, F) → takes last token → linear head.
    """
    def __init__(self, F: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(F, d_model)
        self.pos_enc = _SinusoidalPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F) → (B,)
        x = self.pos_enc(self.input_proj(x))               # (B, W, d_model)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.encoder(x, mask=mask, is_causal=True)     # (B, W, d_model)
        return self.head(x[:, -1, :]).squeeze(-1)           # last token → (B,)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "CNN": CNN1D,
    "LSTM": CausalLSTM,
    "GRU": CausalGRU,
    "TCN": TCN,
    "Transformer": CausalTransformer,
}


def build_model(name: str, W: int, F: int) -> nn.Module:
    if name == "CNN":
        return CNN1D(W, F)
    elif name == "LSTM":
        return CausalLSTM(F)
    elif name == "GRU":
        return CausalGRU(F)
    elif name == "TCN":
        return TCN(F)
    elif name == "Transformer":
        return CausalTransformer(F)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")


# ── data helpers ───────────────────────────────────────────────────────────────

def make_tensor_dataset(X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> TensorDataset:
    N, W, F = X.shape
    X_flat = scaler.transform(X.reshape(N, -1)).reshape(N, W, F).astype(np.float32)
    return TensorDataset(
        torch.from_numpy(X_flat),
        torch.from_numpy(y.astype(np.float32)),
    )


def load_windows(H: int):
    path = PROCESSED_DIR / f"windows_H{H:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run 01_build_anticipatory_dataset.py first.")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"].astype(np.int32), d["circuits"], d["lap_keys"]


def load_scaler(held_out: str, H: int) -> StandardScaler:
    path = PROCESSED_DIR / f"scaler_holdout_{held_out}_H{H:03d}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler: {path}. Run 01_build_anticipatory_dataset.py first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── training ───────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for X_b, y_b in loader:
        logits = model(X_b.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y_b.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def fit_with_early_stopping(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    verbose: bool = False,
) -> tuple[nn.Module, int]:
    """Train with early stopping on val AUC-PR. Returns best model + epoch stopped."""
    set_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0)

    best_auc_pr = 0.0
    best_state: dict = {}
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_prob, val_true = evaluate_loader(model, val_loader, device)

        n_pos = int(val_true.sum())
        val_auc_pr = (average_precision_score(val_true, val_prob)
                      if n_pos > 0 and n_pos < len(val_true) else 0.5)
        scheduler.step(val_auc_pr)

        if val_auc_pr > best_auc_pr + 1e-5:
            best_auc_pr = val_auc_pr
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"      epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"val_AUC-PR={val_auc_pr:.4f}  best={best_auc_pr:.4f}  "
                  f"no-improve={no_improve}")

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, epoch


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    return {
        "n_test": len(y_true),
        "xmode_rate": float(y_true.mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_xmode": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_zmode": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob))
                   if n_pos > 0 and n_neg > 0 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_prob))
                  if n_pos > 0 else float("nan"),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--horizons", type=int, nargs="+", default=[10],
                   help="Horizons to train on (default: 10). For full sweep: 1 5 10 25 50.")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                   choices=list(MODEL_REGISTRY),
                   help="Models to train (default: all).")
    p.add_argument("--max-epochs", type=int, default=80,
                   help="Max training epochs (default: 80).")
    p.add_argument("--patience", type=int, default=12,
                   help="Early stopping patience (default: 12).")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Training batch size (default: 512).")
    p.add_argument("--fast", action="store_true",
                   help="Quick smoke-test: 10 epochs, no early stopping.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-epoch training progress.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Override data directory (default: multi_circuit_work/processed/). "
                        "Use on Colab: --data-dir /content/processed")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output directory for results/models. Default: same as data-dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    # allow --data-dir override for Colab / remote runs
    global PROCESSED_DIR, PREDS_DIR, MODELS_DIR
    if args.data_dir is not None:
        PROCESSED_DIR = args.data_dir
    out_base = args.out_dir or PROCESSED_DIR
    PREDS_DIR = out_base / "deep_preds"
    MODELS_DIR = out_base / "models"

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.fast:
        args.max_epochs = 10
        args.patience = 99  # no early stopping in fast mode
        print("Fast mode: 10 epochs, no early stopping.")

    all_rows: list[dict] = []

    for H in args.horizons:
        print(f"\n{'=' * 64}")
        print(f"H = {H} (~{H/10:.1f} s ahead)")

        X, y, circuits, lap_keys = load_windows(H)
        N, W, F = X.shape
        circuit_list = sorted(set(circuits.tolist()))

        if len(circuit_list) < 2:
            print(f"  Only 1 circuit — LOCO requires ≥2. Skipping H={H}.")
            continue

        for held_out in circuit_list:
            test_mask = circuits == held_out
            train_mask = ~test_mask
            n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
            print(f"\n  Fold hold-out={held_out}  train={n_tr:,}  test={n_te:,}")

            scaler = load_scaler(held_out, H)

            # val split: stratified 10% of train windows
            tr_idx = np.where(train_mask)[0]
            y_tr = y[train_mask]
            try:
                idx_tr, idx_val = train_test_split(
                    tr_idx, test_size=0.10, stratify=y_tr, random_state=SEED
                )
            except ValueError:
                # fallback if too few positives for stratified split
                idx_tr, idx_val = train_test_split(tr_idx, test_size=0.10, random_state=SEED)

            train_ds = make_tensor_dataset(X[idx_tr], y[idx_tr], scaler)
            val_ds = make_tensor_dataset(X[idx_val], y[idx_val], scaler)
            test_ds = make_tensor_dataset(X[test_mask], y[test_mask], scaler)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2,
                                     shuffle=False, num_workers=0)

            # focal loss alpha: inverse-frequency of positive class in train
            pos_rate = float(y[idx_tr].mean())
            focal_alpha = 1.0 - pos_rate   # downweight majority class
            criterion = FocalLoss(alpha=focal_alpha, gamma=2.0).to(device)

            for model_name in args.models:
                print(f"    [{model_name}] ...", end=" ", flush=True)
                set_seed(SEED)
                model = build_model(model_name, W, F).to(device)

                model, stopped_epoch = fit_with_early_stopping(
                    model, train_ds, val_ds, criterion, device,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    verbose=args.verbose,
                )

                y_prob, y_true_eval = evaluate_loader(model, test_loader, device)
                y_pred = (y_prob >= 0.5).astype(np.int32)
                m = compute_metrics(y_true_eval, y_prob, y_pred)

                print(f"ep={stopped_epoch:3d}  "
                      f"F1={m['f1_macro']:.3f}  "
                      f"AUC-ROC={m['auc_roc']:.3f}  "
                      f"AUC-PR={m['auc_pr']:.3f}")

                row = {"model": model_name, "held_out": held_out, "H": H,
                       "n_train": n_tr, "stopped_epoch": stopped_epoch, **m}
                all_rows.append(row)

                # save predictions
                np.savez_compressed(
                    PREDS_DIR / f"{model_name}_{held_out}_H{H:03d}.npz",
                    y_prob=y_prob, y_true=y_true_eval, y_pred=y_pred,
                )
                # save model checkpoint
                ckpt_path = MODELS_DIR / f"{model_name}_fold_{held_out}_H{H:03d}.pt"
                torch.save({"model_state": model.state_dict(),
                            "W": W, "F": F, "H": H, "held_out": held_out,
                            "epoch": stopped_epoch}, ckpt_path)

    if not all_rows:
        print("\nNo results. Check that ≥2 circuit CSVs exist and --horizons are valid.")
        return

    results = pd.DataFrame(all_rows)
    out_path = PROCESSED_DIR / "deep_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    print("\n── Deep model summary (mean across LOCO folds) ──────────────────")
    summary = (
        results
        .groupby(["model", "H"])[["f1_macro", "f1_xmode", "auc_roc", "auc_pr"]]
        .mean()
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
