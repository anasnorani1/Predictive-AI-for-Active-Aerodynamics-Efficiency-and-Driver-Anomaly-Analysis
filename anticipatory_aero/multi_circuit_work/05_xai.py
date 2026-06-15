"""
05_xai.py
==========
Interpretability analyses for the anticipatory aero prediction models.

Methods:
  1. SHAP TreeExplainer — for RF-lag (fast, exact Shapley values)
  2. Integrated Gradients — for any PyTorch model (CNN / LSTM / GRU / TCN / Transformer)
  3. Attention weight extraction — for CausalTransformer
  4. Temporal SHAP heatmap — average |IG| per time step × feature for CNN/TCN

Usage:
  cd multi_circuit_work
  python 05_xai.py --H 10 --held-out Monza          # RF-lag SHAP + IG for best model
  python 05_xai.py --H 10 --held-out Monza --model Transformer  # attention maps

Outputs:
  graphs/shap_rflag_H{H:03d}_{held_out}.pdf
  graphs/ig_heatmap_{model}_H{H:03d}_{held_out}.pdf
  graphs/attention_{held_out}_H{H:03d}.pdf
  processed/ig_{model}_{held_out}_H{H:03d}.npy   (mean |IG| per feature per time step)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

WORK_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = WORK_ROOT / "processed"
MODELS_DIR = WORK_ROOT / "models"
GRAPHS_DIR = WORK_ROOT / "graphs"

FEATURE_COLS = [
    "Speed", "RPM", "nGear", "Throttle", "Brake",
    "X", "Y", "Z",
    "Acceleration", "Elevation_Delta",
    "Kinetic_Energy_MJ", "Longitudinal_Force_N",
]


# ── data helpers ──────────────────────────────────────────────────────────────

def load_windows(H: int):
    p = PROCESSED_DIR / f"windows_H{H:03d}.npz"
    d = np.load(p, allow_pickle=True)
    return d["X"], d["y"].astype(np.int32), d["circuits"]


def load_scaler(held_out: str, H: int):
    p = PROCESSED_DIR / f"scaler_holdout_{held_out}_H{H:03d}.pkl"
    with open(p, "rb") as fh:
        return pickle.load(fh)


def load_torch_model(model_name: str, held_out: str, H: int, W: int, F: int):
    """Load a trained PyTorch model from checkpoint."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deep", WORK_ROOT / "03_deep_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt_path = MODELS_DIR / f"{model_name}_fold_{held_out}_H{H:03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}. Run 03_deep_models.py first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mod.build_model(model_name, W, F)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── SHAP for RF-lag ───────────────────────────────────────────────────────────

def shap_rf_lag(H: int, held_out: str, n_bg: int = 200, n_explain: int = 500) -> None:
    if not HAS_SHAP:
        print("shap not installed — skipping RF-lag SHAP. Run: pip install shap")
        return
    if not HAS_MPL:
        print("matplotlib not available — skipping SHAP plot.")
        return

    # Load saved RF-lag model (pkl via sklearn)
    rf_path = MODELS_DIR / f"RF-lag_fold_{held_out}_H{H:03d}_model.pkl"
    if not rf_path.exists():
        # Retrain quickly if needed
        print(f"  RF-lag checkpoint not found at {rf_path}. Skipping SHAP.")
        print(f"  (Run 02_baselines.py with --save-models flag to save RF models.)")
        return

    with open(rf_path, "rb") as fh:
        clf = pickle.load(fh)

    X, y, circuits = load_windows(H)
    N, W, F = X.shape
    scaler = load_scaler(held_out, H)

    test_mask = circuits == held_out
    train_mask = ~test_mask

    X_tr = scaler.transform(X[train_mask].reshape(-1, W * F))
    X_te = scaler.transform(X[test_mask].reshape(-1, W * F))

    # sample background and explanation sets
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_tr), min(n_bg, len(X_tr)), replace=False)
    ex_idx = rng.choice(len(X_te), min(n_explain, len(X_te)), replace=False)

    explainer = shap.TreeExplainer(clf, data=X_tr[bg_idx], feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_te[ex_idx])
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class-1 (X-mode) SHAP values
    else:
        sv = shap_values

    # reshape to (n_explain, W, F) and average over time steps
    sv_reshaped = sv.reshape(-1, W, len(FEATURE_COLS))
    mean_abs_sv = np.abs(sv_reshaped).mean(axis=(0, 1))   # (F,) — per feature

    GRAPHS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(mean_abs_sv)[::-1]
    ax.barh([FEATURE_COLS[i] for i in order], mean_abs_sv[order], color="steelblue")
    ax.set_xlabel("|SHAP| value (mean absolute)", fontsize=11)
    ax.set_title(f"RF-lag Feature Importance (SHAP)\nH={H}, test circuit={held_out}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = GRAPHS_DIR / f"shap_rflag_H{H:03d}_{held_out}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP plot saved -> {path.name}")


# ── Integrated Gradients ──────────────────────────────────────────────────────

def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,         # (N, W, F)
    baseline: torch.Tensor,  # (1, W, F) — typically zeros
    n_steps: int = 50,
) -> torch.Tensor:
    """
    IG attribution: (x - baseline) × mean(grad_F w.r.t. interpolated inputs).
    Returns attributions of shape (N, W, F).
    """
    x.requires_grad_(False)
    baseline.requires_grad_(False)
    N, W, F_ = x.shape
    attributions = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, n_steps + 1, device=x.device)
    for alpha in alphas:
        interp = baseline + alpha * (x - baseline)  # (N, W, F)
        interp = interp.detach().requires_grad_(True)
        logits = model(interp)
        grads = torch.autograd.grad(logits.sum(), interp)[0]  # (N, W, F)
        attributions += grads.detach()

    attributions /= (n_steps + 1)
    attributions *= (x - baseline)
    return attributions   # (N, W, F)


def run_ig(model_name: str, H: int, held_out: str,
           n_samples: int = 300, n_steps: int = 50) -> None:
    if not HAS_MPL:
        print("matplotlib not available — skipping IG plot.")
        return

    X, y, circuits = load_windows(H)
    N, W, F = X.shape
    scaler = load_scaler(held_out, H)
    test_mask = circuits == held_out

    model = load_torch_model(model_name, held_out, H, W, F)

    X_te_raw = X[test_mask]
    y_te = y[test_mask]
    X_te_sc = scaler.transform(X_te_raw.reshape(-1, W * F)).reshape(-1, W, F).astype(np.float32)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te_sc), min(n_samples, len(X_te_sc)), replace=False)
    X_batch = torch.from_numpy(X_te_sc[idx])
    baseline = torch.zeros(1, W, F)

    model.train()  # enable grad for some model types
    attrs = integrated_gradients(model, X_batch, baseline, n_steps).numpy()  # (n, W, F)

    # mean absolute attribution per (time step, feature)
    mean_abs = np.abs(attrs).mean(axis=0)   # (W, F)

    GRAPHS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # heatmap: time × feature
    im = axes[0].imshow(mean_abs.T, aspect="auto", cmap="hot_r",
                        origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=axes[0], label="Mean |IG|")
    axes[0].set_xlabel("Time step in window", fontsize=10)
    axes[0].set_yticks(range(len(FEATURE_COLS)))
    axes[0].set_yticklabels(FEATURE_COLS, fontsize=8)
    axes[0].set_title(f"{model_name} — Integrated Gradients\nH={H}, test={held_out}", fontsize=10)

    # bar: per-feature (averaged over time)
    feat_mean = mean_abs.mean(axis=0)   # (F,)
    order = np.argsort(feat_mean)[::-1]
    axes[1].barh([FEATURE_COLS[i] for i in order], feat_mean[order], color="tomato")
    axes[1].set_xlabel("Mean |IG| (avg over time steps)", fontsize=10)
    axes[1].set_title("Feature importance (IG)", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = GRAPHS_DIR / f"ig_{model_name}_H{H:03d}_{held_out}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(PROCESSED_DIR / f"ig_{model_name}_{held_out}_H{H:03d}.npy", mean_abs)
    print(f"  IG heatmap saved -> {path.name}")


# ── Transformer attention maps ────────────────────────────────────────────────

def extract_attention_weights(
    model,
    X_batch: torch.Tensor,
) -> np.ndarray:
    """
    Extract causal-Transformer attention weights layer by layer.
    PyTorch's TransformerEncoderLayer._sa_block hardcodes need_weights=False,
    so hooks return None. Instead we run each layer manually with need_weights=True.
    Returns averaged attention (W, W) across layers and batch samples.
    """
    import math

    model.eval()
    W_seq = X_batch.size(1)

    # Build causal mask (same as CausalTransformer.forward)
    causal_mask = torch.triu(
        torch.ones(W_seq, W_seq), diagonal=1
    ).masked_fill(torch.triu(torch.ones(W_seq, W_seq), diagonal=1) == 1, float("-inf"))

    all_attn: list[torch.Tensor] = []

    with torch.no_grad():
        # Run the input projection + positional encoding first
        x = model.pos_enc(model.input_proj(X_batch))   # (B, W, d_model)

        for layer in model.encoder.layers:
            # Extract attention with need_weights=True by calling self_attn directly
            # layer.norm1 is pre-norm in some configs; ours is post-norm (default)
            x_norm = layer.norm1(x)

            attn_out, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=causal_mask,
                need_weights=True,
                average_attn_weights=True,   # (B, W, W) not (B, heads, W, W)
            )
            all_attn.append(attn_w.cpu())   # (B, W, W)

            # Continue the forward pass through this layer normally
            x = layer(x, src_mask=causal_mask, is_causal=True)

    if not all_attn:
        return None

    # Stack and average over layers and batch: (n_layers, B, W, W) -> (W, W)
    stacked = torch.stack(all_attn, dim=0)   # (n_layers, B, W, W)
    return stacked.mean(dim=(0, 1)).numpy()  # (W, W)


def run_attention(H: int, held_out: str, n_samples: int = 100) -> None:
    if not HAS_MPL:
        return

    X, y, circuits = load_windows(H)
    N, W, F = X.shape
    scaler = load_scaler(held_out, H)
    test_mask = circuits == held_out

    try:
        model = load_torch_model("Transformer", held_out, H, W, F)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    X_te_raw = X[test_mask]
    X_te_sc = scaler.transform(X_te_raw.reshape(-1, W * F)).reshape(-1, W, F).astype(np.float32)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te_sc), min(n_samples, len(X_te_sc)), replace=False)
    X_batch = torch.from_numpy(X_te_sc[idx])

    avg_attn = extract_attention_weights(model, X_batch)
    if avg_attn is None:
        print("  Could not extract attention weights.")
        return

    GRAPHS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_attn, aspect="auto", cmap="Blues", origin="upper")
    plt.colorbar(im, ax=ax, label="Avg attention weight")
    ax.set_xlabel("Key time step", fontsize=10)
    ax.set_ylabel("Query time step", fontsize=10)
    ax.set_title(f"Transformer attention (mean over samples & layers)\n"
                 f"H={H}, test={held_out}", fontsize=10)
    fig.tight_layout()
    path = GRAPHS_DIR / f"attention_H{H:03d}_{held_out}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Attention map saved -> {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--H", type=int, default=10, help="Horizon to analyse.")
    p.add_argument("--held-out", type=str, default=None,
                   help="Circuit to use as the test fold. Default: first available.")
    p.add_argument("--model", type=str, default="TCN",
                   choices=["CNN", "LSTM", "GRU", "TCN", "Transformer"],
                   help="Deep model for IG analysis.")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-ig", action="store_true")
    p.add_argument("--skip-attention", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    H = args.H

    X, y, circuits = load_windows(H)
    available = sorted(set(circuits.tolist()))
    held_out = args.held_out or available[0]
    if held_out not in available:
        raise ValueError(f"--held-out {held_out} not in {available}")

    print(f"XAI analysis: H={H}, test circuit={held_out}")

    if not args.skip_shap:
        print("\n1. SHAP (RF-lag) ...")
        shap_rf_lag(H, held_out)

    if not args.skip_ig:
        print(f"\n2. Integrated Gradients ({args.model}) ...")
        try:
            run_ig(args.model, H, held_out)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")

    if not args.skip_attention:
        print("\n3. Transformer attention maps ...")
        run_attention(H, held_out)


if __name__ == "__main__":
    main()
