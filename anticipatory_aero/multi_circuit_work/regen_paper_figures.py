"""
regen_paper_figures.py
======================
Regenerate the 5 Phase-2 figures that changed after the clean Kaggle
reproducibility run, writing them directly into docs/figures/ so they
match the updated tables/text in IEEE_Paper.tex.

Figures produced:
  plot_loco_heatmap.png        (per-circuit AUC; no more GRU/TCN "collapse")
  plot_model_comparison_H10.png(10-model mean AUC with std)
  plot_horizon_sweep.png       (AUC vs horizon)
  plot_transition_f1.png       (stable vs transition F1)
  ig_GRU_H010_Monaco.png       (new IG attributions)
  ig_Transformer_H010_Monaco.png

Run:  python regen_paper_figures.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
# paper uses figures/ relative to docs/  -> repo_root/docs/figures
FIGDIR = HERE.parents[1] / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CIRCUITS = ["Monaco", "Monza", "Silverstone", "Suzuka"]
ORDER = ["LR-instant", "RF-instant", "XGBoost-instant", "RF-lag",
         "CausalTrans.", "CausalGRU", "CausalLSTM", "TCN", "CNN1D", "Persistence"]
# map CSV model names -> display names
NAMEMAP = {"Transformer": "CausalTrans.", "GRU": "CausalGRU", "LSTM": "CausalLSTM",
           "CNN": "CNN1D"}
PALETTE = {
    "LR-instant": "#2ecc71", "RF-instant": "#27ae60", "XGBoost-instant": "#16a085",
    "RF-lag": "#f39c12", "CausalTrans.": "#3498db", "CausalGRU": "#e74c3c",
    "CausalLSTM": "#9b59b6", "TCN": "#1abc9c", "CNN1D": "#95a5a6",
    "Persistence": "#7f8c8d",
}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load():
    df = pd.read_csv(PROC / "ALL_results_combined.csv")
    df["model"] = df["model"].replace(NAMEMAP)
    return df


# ── 1. Model comparison bar (mean AUC +/- std, H=10) ────────────────────────────
def fig_model_comparison(df):
    h = df[df.H == 10]
    means = h.groupby("model")["auc_roc"].mean()
    stds = h.groupby("model")["auc_roc"].std()
    models = [m for m in ORDER if m in means.index]
    vals = [means[m] for m in models]
    errs = [stds[m] for m in models]
    cols = [PALETTE[m] for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(models)), vals, yerr=errs, capsize=4,
                  color=cols, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("AUC-ROC (mean $\\pm$ std across 4 LOCO folds)")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Model Performance at $H{=}10$ (1-second horizon)", fontweight="bold")
    ax.axhline(means["LR-instant"], ls="--", c="#2ecc71", lw=1, alpha=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i] + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGDIR / "plot_model_comparison_H10.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  plot_model_comparison_H10.png")


# ── 2. LOCO heatmap (per-circuit AUC) ──────────────────────────────────────────
def fig_heatmap(df):
    h = df[df.H == 10]
    models = [m for m in ORDER if m in set(h.model)]
    M = np.array([[h[(h.model == m) & (h.held_out == c)]["auc_roc"].mean()
                   for c in CIRCUITS] for m in models])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CIRCUITS))); ax.set_xticklabels(CIRCUITS)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(CIRCUITS)):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=8.5)
    ax.set_title("LOCO AUC-ROC by Model and Circuit ($H{=}10$)", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("AUC-ROC")
    fig.tight_layout()
    fig.savefig(FIGDIR / "plot_loco_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  plot_loco_heatmap.png")


# ── 3. Horizon sweep ───────────────────────────────────────────────────────────
def fig_horizon(df):
    Hs = [1, 5, 10, 25, 50]
    show = ["LR-instant", "RF-instant", "RF-lag", "CausalTrans.", "CausalGRU"]
    # deep models only have fresh H=10; use available horizons present in df
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in show:
        sub = df[df.model == m]
        ys = [sub[sub.H == H]["auc_roc"].mean() for H in Hs]
        ax.plot(Hs, ys, "o-", color=PALETTE[m], label=m, lw=1.8, ms=5)
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xlabel("Prediction horizon $H$ (frames; $H{=}10 \\approx 1$\\,s)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC vs. Prediction Horizon", fontweight="bold")
    ax.set_xticks(Hs); ax.legend(fontsize=9); ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGDIR / "plot_horizon_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  plot_horizon_sweep.png")


# ── 4. Transition vs stable F1 (values from clean-run computation) ─────────────
def fig_transition():
    # stable, transition per circuit (computed from clean-run predictions)
    data = {
        "CausalTrans.": {"Monaco": (0.934, 0.628), "Monza": (0.891, 0.624),
                         "Silverstone": (0.959, 0.675), "Suzuka": (0.801, 0.598)},
        "CausalGRU":    {"Monaco": (0.722, 0.550), "Monza": (0.952, 0.605),
                         "Silverstone": (0.973, 0.687), "Suzuka": (0.806, 0.593)},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    x = np.arange(len(CIRCUITS)); w = 0.36
    for ax, (model, d) in zip(axes, data.items()):
        stable = [d[c][0] for c in CIRCUITS]
        trans = [d[c][1] for c in CIRCUITS]
        ax.bar(x - w/2, stable, w, label="Stable", color="#3498db", edgecolor="black", lw=0.5)
        ax.bar(x + w/2, trans, w, label="Transition", color="#e67e22", edgecolor="black", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(CIRCUITS, rotation=20, ha="right")
        ax.set_title(model, fontweight="bold"); ax.set_ylim(0, 1.0)
        ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].set_ylabel("F1-Macro"); axes[0].legend(fontsize=9)
    fig.suptitle("F1-Macro at Stable vs.\\ Transition Windows ($\\pm$0.5\\,s)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "plot_transition_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("  plot_transition_f1.png")


# ── 5. IG attribution bars (GRU, Transformer) ──────────────────────────────────
def fig_ig():
    ig = pd.read_csv(PROC / "ig_feature_attributions.csv", index_col=0)
    for model, color in [("GRU", "#e74c3c"), ("Transformer", "#3498db")]:
        s = ig[model].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.barh(range(len(s)), s.values, color=color, edgecolor="black", lw=0.5)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel("Mean |Integrated Gradients attribution|")
        disp = "CausalGRU" if model == "GRU" else "CausalTransformer"
        ax.set_title(f"{disp} — IG at Monaco ($H{{=}}10$)", fontweight="bold", fontsize=11)
        for i, v in enumerate(s.values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, max(s.values) * 1.18); ax.grid(axis="x", ls=":", alpha=0.4)
        fig.tight_layout()
        fname = f"ig_{model}_H010_Monaco.png"
        fig.savefig(FIGDIR / fname, dpi=160, bbox_inches="tight")
        plt.close(fig); print(f"  {fname}")


def main():
    print(f"Writing figures -> {FIGDIR}")
    df = load()
    fig_model_comparison(df)
    fig_heatmap(df)
    fig_horizon(df)
    fig_transition()
    fig_ig()
    print("Done. Regenerated 6 PNG files.")


if __name__ == "__main__":
    main()
