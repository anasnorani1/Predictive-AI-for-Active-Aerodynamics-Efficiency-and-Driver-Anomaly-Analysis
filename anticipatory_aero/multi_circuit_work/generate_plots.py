"""
Generate all summary result plots for the project dashboard.
Run with Python 3.10: python generate_plots.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

GRAPHS = Path(__file__).parent / "graphs"
RESULTS = Path(__file__).parent.parent / "Results"
GRAPHS.mkdir(exist_ok=True)

# Colour palette
C = {
    "LR-instant":  "#2ecc71",
    "RF-instant":  "#27ae60",
    "RF-lag":      "#f39c12",
    "Transformer": "#3498db",
    "LSTM":        "#9b59b6",
    "GRU":         "#e74c3c",
    "TCN":         "#1abc9c",
    "CNN":         "#95a5a6",
}
CIRCUITS = ["Monaco", "Monza", "Silverstone", "Suzuka"]
MODELS_ORDER = ["LR-instant","RF-instant","RF-lag","Transformer","LSTM","GRU","TCN","CNN"]

bl = pd.read_csv(RESULTS / "baseline_results.csv")
dl = pd.read_csv(RESULTS / "deep_results.csv")
sw = pd.read_csv(RESULTS / "sweep_results.csv")
h10 = pd.concat([bl[bl.H==10], dl], ignore_index=True)

# ── 1. Model comparison at H=10 (AUC-ROC + F1) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Performance at H=10 (1-second horizon) — All 4 LOCO Folds",
             fontsize=13, fontweight="bold", y=1.01)

for ax, metric, label in zip(axes, ["auc_roc","f1_macro"], ["AUC-ROC","F1-Macro"]):
    means = h10.groupby("model")[metric].mean().reindex(MODELS_ORDER)
    stds  = h10.groupby("model")[metric].std().reindex(MODELS_ORDER)
    colors = [C[m] for m in MODELS_ORDER]
    bars = ax.barh(MODELS_ORDER, means, xerr=stds, color=colors, edgecolor="white",
                   linewidth=0.8, error_kw=dict(elinewidth=1.2, ecolor="#555", capsize=3))
    ax.axvline(means["LR-instant"], color="black", linestyle="--", linewidth=1, alpha=0.4)
    for i, (m, v, s) in enumerate(zip(MODELS_ORDER, means, stds)):
        ax.text(v + s + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5, color="#333")
    ax.set_xlim(0.6, 1.02)
    ax.set_xlabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    # shade deep learning region
    ax.axhspan(2.5, 7.5, color="#f0f0f0", zorder=0, alpha=0.5)
    ax.text(0.61, 3, "Deep\nLearning", va="center", fontsize=8, color="#aaa", style="italic")

deep_patch = mpatches.Patch(facecolor="#f0f0f0", edgecolor="#ccc", label="Deep learning models")
fig.legend(handles=[deep_patch], loc="lower center", ncol=1, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
fig.savefig(GRAPHS / "plot_model_comparison_H10.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_model_comparison_H10.png")

# ── 2. Horizon sweep ─────────────────────────────────────────────────────────
sweep_bl = bl.groupby(["model","H"])["auc_roc"].mean().unstack("H")
sweep_dl = sw.groupby(["model","H"])["auc_roc"].mean().unstack("H")
sweep_all = pd.concat([sweep_bl, sweep_dl])
Hs = [1, 5, 10, 25, 50]
labels = ["0.1s","0.5s","1s","2.5s","5s"]

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("AUC-ROC vs Prediction Horizon — Mean over 4 LOCO Folds",
             fontsize=13, fontweight="bold")
for m in ["LR-instant","RF-instant","RF-lag","Transformer","GRU"]:
    if m not in sweep_all.index: continue
    vals = sweep_all.loc[m, Hs]
    lw = 2.5 if m in ("LR-instant","Transformer") else 1.5
    ls = "-" if m in ("LR-instant","Transformer","GRU") else "--"
    ax.plot(range(len(Hs)), vals, marker="o", label=m, color=C[m],
            linewidth=lw, linestyle=ls, markersize=6)

ax.axvspan(1.5, 2.5, color="#ffe0e0", alpha=0.25, zorder=0)
ax.text(2.0, 0.52, "Meaningful\nanticipation\nregime", ha="center",
        fontsize=8, color="#c0392b", style="italic")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.text(4.1, 0.51, "Chance", fontsize=8, color="gray")

ax.set_xticks(range(len(Hs)))
ax.set_xticklabels([f"H={h}\n({l})" for h,l in zip(Hs,labels)], fontsize=9)
ax.set_ylabel("Mean AUC-ROC", fontsize=11)
ax.set_ylim(0.40, 1.02)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAPHS / "plot_horizon_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_horizon_sweep.png")

# ── 3. LOCO heatmap ──────────────────────────────────────────────────────────
pivot = h10.pivot_table(values="auc_roc", index="model", columns="held_out") \
           .reindex(index=MODELS_ORDER, columns=CIRCUITS)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("AUC-ROC Heatmap: All Models × All Circuits at H=10",
             fontsize=13, fontweight="bold")
import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn
im = ax.imshow(pivot.values, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, label="AUC-ROC", shrink=0.8)

ax.set_xticks(range(len(CIRCUITS))); ax.set_xticklabels(CIRCUITS, fontsize=10)
ax.set_yticks(range(len(MODELS_ORDER))); ax.set_yticklabels(MODELS_ORDER, fontsize=10)

for i, model in enumerate(MODELS_ORDER):
    for j, circuit in enumerate(CIRCUITS):
        val = pivot.loc[model, circuit]
        txt_color = "white" if val < 0.75 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=txt_color)

# Highlight two collapse cells
for (mi, ci), label in [((5, 0), "COLLAPSE"), ((6, 3), "COLLAPSE")]:
    ax.add_patch(plt.Rectangle((ci-0.5, mi-0.5), 1, 1,
                 fill=False, edgecolor="red", linewidth=2.5))

ax.set_xlabel("Test Circuit (held-out)", fontsize=11)
ax.set_ylabel("Model", fontsize=11)

# Divider between classical and deep
ax.axhline(2.5, color="white", linewidth=2)
ax.text(3.55, 2.5, "Classical | Deep", fontsize=7.5, color="white",
        va="center", ha="right", rotation=90)

fig.tight_layout()
fig.savefig(GRAPHS / "plot_loco_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_loco_heatmap.png")

# ── 4. Monaco collapse — all models ranked ───────────────────────────────────
monaco = h10[h10.held_out == "Monaco"].set_index("model") \
             .reindex(MODELS_ORDER)[["auc_roc","f1_xmode"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Monaco Fold Performance (H=10) — The Hardest Generalisation Test",
             fontsize=13, fontweight="bold")

for ax, col, label in zip(axes, ["auc_roc","f1_xmode"], ["AUC-ROC","X-Mode F1"]):
    colors = [C[m] for m in MODELS_ORDER]
    bars = ax.bar(range(len(MODELS_ORDER)), monaco[col], color=colors,
                  edgecolor="white", linewidth=0.8)
    for i, (m, v) in enumerate(zip(MODELS_ORDER, monaco[col])):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.text(7.4, 0.51, "Chance", fontsize=8, color="gray", ha="right")
    ax.set_xticks(range(len(MODELS_ORDER)))
    ax.set_xticklabels(MODELS_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"Monaco — {label}", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    # Annotate GRU bar
    gru_i = MODELS_ORDER.index("GRU")
    ax.annotate("GRU\ncollapse", xy=(gru_i, monaco.loc["GRU", col]),
                xytext=(gru_i - 1.5, 0.35), fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

fig.tight_layout()
fig.savefig(GRAPHS / "plot_monaco_collapse.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_monaco_collapse.png")

# ── 5. Transition-window F1 ──────────────────────────────────────────────────
trans_data = {
    "Transformer": {"Monaco":   (0.496, 0.902), "Monza":       (0.551, 0.865),
                    "Silverstone": (0.659, 0.970), "Suzuka":    (0.554, 0.599)},
    "GRU":         {"Monaco":   (0.328, 0.480), "Monza":       (0.615, 0.943),
                    "Silverstone": (0.674, 0.973), "Suzuka":    (0.575, 0.860)},
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Transition-Window vs Stable-Window F1-Macro at H=10 (±0.5s around mode changes)",
             fontsize=12, fontweight="bold")

for ax, model_name in zip(axes, ["Transformer", "GRU"]):
    trans_vals  = [trans_data[model_name][c][0] for c in CIRCUITS]
    stable_vals = [trans_data[model_name][c][1] for c in CIRCUITS]
    x = np.arange(len(CIRCUITS))
    width = 0.35
    b1 = ax.bar(x - width/2, trans_vals,  width, label="Transition windows", color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x + width/2, stable_vals, width, label="Stable windows",     color="#3498db", alpha=0.85)
    for rect in b1: ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.01,
                            f"{rect.get_height():.2f}", ha="center", fontsize=8)
    for rect in b2: ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.01,
                            f"{rect.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CIRCUITS, fontsize=10)
    ax.set_ylabel("F1-Macro", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{model_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    mt = np.mean(trans_vals); ms = np.mean(stable_vals)
    ax.text(0.02, 0.97, f"Mean trans={mt:.3f}  stable={ms:.3f}  Δ={mt-ms:+.3f}",
            transform=ax.transAxes, fontsize=8, va="top", color="#555")

fig.tight_layout()
fig.savefig(GRAPHS / "plot_transition_f1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_transition_f1.png")

# ── 6. Feature importance comparison — GRU vs Transformer at Monaco ───────────
FEATURE_COLS = ["Speed","RPM","nGear","Throttle","Brake","X","Y","Z",
                "Acceleration","Elevation_Delta","Kinetic_Energy_MJ","Longitudinal_Force_N"]
PROCESSED = Path(__file__).parent / "processed"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Integrated Gradients Feature Importance — Monaco Fold H=10\n"
             "(Why GRU fails: it focuses on speed-dynamics that are circuit-calibrated)",
             fontsize=11, fontweight="bold")

for ax, model_name, color in zip(axes, ["GRU","Transformer"], ["#e74c3c","#3498db"]):
    p = PROCESSED / f"ig_{model_name}_Monaco_H010.npy"
    if not p.exists():
        ax.set_title(f"{model_name} — file missing"); continue
    arr = np.load(p)
    feat_mean = arr.mean(axis=0)
    total = feat_mean.sum()
    order = np.argsort(feat_mean)
    pcts = 100 * feat_mean[order] / total
    labels = [FEATURE_COLS[i] for i in order]
    bars = ax.barh(labels, pcts, color=color, alpha=0.85, edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 0.2, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of total |IG| attribution", fontsize=10)
    ax.set_title(f"{model_name} — Monaco", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    # Annotate top feature
    top_label = FEATURE_COLS[order[-1]]
    ax.annotate(f"Top: {top_label}", xy=(pcts[-1], len(pcts)-1),
                xytext=(pcts[-1]-6, len(pcts)-3),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333", lw=1))

fig.tight_layout()
fig.savefig(GRAPHS / "plot_feature_importance_monaco.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_feature_importance_monaco.png")

# ── 7. GRU full horizon per circuit (showing Monaco collapse trajectory) ───────
gru_sw = sw[sw.model == "GRU"].pivot_table(values="auc_roc", index="H", columns="held_out")
tr_sw  = sw[sw.model == "Transformer"].pivot_table(values="auc_roc", index="H", columns="held_out")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("AUC-ROC vs Horizon Per Circuit — GRU (collapse at Monaco) vs Transformer (robust)",
             fontsize=12, fontweight="bold")

circuit_colors = {"Monaco":"#e74c3c","Monza":"#3498db","Silverstone":"#2ecc71","Suzuka":"#f39c12"}
for ax, df, title in zip(axes, [gru_sw, tr_sw], ["GRU","Transformer"]):
    for c in CIRCUITS:
        if c not in df.columns: continue
        lw = 2.5 if c == "Monaco" else 1.5
        ax.plot(Hs, df.loc[Hs, c], marker="o", label=c,
                color=circuit_colors[c], linewidth=lw, markersize=5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks(Hs); ax.set_xticklabels([f"H={h}" for h in Hs])
    ax.set_ylabel("AUC-ROC", fontsize=10)
    ax.set_ylim(0.2, 1.03)
    ax.set_title(f"{title} — Per-Circuit AUC-ROC", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if title == "GRU":
        ax.annotate("Monaco\ncollapse\nat H=10", xy=(10, gru_sw.loc[10,"Monaco"]),
                    xytext=(15, 0.60),
                    fontsize=8.5, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.3))

fig.tight_layout()
fig.savefig(GRAPHS / "plot_gru_vs_transformer_horizon.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_gru_vs_transformer_horizon.png")

print(f"\nAll plots saved to: {GRAPHS}")
print(f"Total graphs: {len(list(GRAPHS.glob('*.png')))} PNG files")
