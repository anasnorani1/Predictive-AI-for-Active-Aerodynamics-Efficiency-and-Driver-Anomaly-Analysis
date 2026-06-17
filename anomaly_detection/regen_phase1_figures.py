"""
regen_phase1_figures.py
=======================
Regenerate the Phase-1 data figures on a clean WHITE background (IEEE print-ready),
replacing the dark dashboard-styled versions. Writes straight into docs/figures/.

Figures: kmeans_elbow, anomaly_track_map, isolation_forest_results,
         roc_curves, stat_08_permutation_importance.

The two *diagrams* (architecture.png, methodology_workflow.png) are hand-drawn and
are NOT regenerated here.

Run:  python regen_phase1_figures.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
ART  = HERE / "artefacts"
FIG  = HERE.parent / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.edgecolor": "#333333", "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
})
BLUE, RED, GREEN = "#1f5fa8", "#d62728", "#2ca02c"

df   = pd.read_csv(ART / "df_final_with_anomalies.csv")
cols = pickle.load(open(ART / "feature_cols.pkl", "rb"))
Xtr  = np.load(ART / "X_train.npy"); ytr = np.load(ART / "y_train.npy")
Xte  = np.load(ART / "X_test.npy");  yte = np.load(ART / "y_test.npy")


# ── 1. K-Means elbow ─────────────────────────────────────────────────────────────
def fig_elbow():
    Xs = StandardScaler().fit_transform(df[cols].values)
    ks = list(range(1, 10)); inertia = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, inertia, "o-", color=BLUE, lw=2, ms=7)
    ax.axvline(4, ls="--", color=RED, lw=1.8, label="$k=4$ (selected)")
    ax.set_xlabel("Number of clusters $k$"); ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("K-Means Elbow — Optimal $k$ Selection", fontweight="bold")
    ax.set_xticks(ks); ax.legend(frameon=True)
    fig.tight_layout(); fig.savefig(FIG / "kmeans_elbow.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  kmeans_elbow.png")


# ── 2. Anomaly track map ─────────────────────────────────────────────────────────
def fig_track_map():
    norm = df[df["Is_Anomaly"] == 0]; ano = df[df["Is_Anomaly"] == 1]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(norm["X"], norm["Y"], s=2, c="#9ecae1", alpha=0.5, label="Normal", rasterized=True)
    ax.scatter(ano["X"],  ano["Y"],  s=6, c=RED, alpha=0.8, label="Anomalous", rasterized=True)
    ax.set_xlabel("Track $X$ (m)"); ax.set_ylabel("Track $Y$ (m)")
    ax.set_title("Spatial Distribution of Anomalies — Suzuka", fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(markerscale=3, frameon=True, loc="best")
    ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "anomaly_track_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  anomaly_track_map.png")


# ── 3. Isolation Forest in Speed–Gear subspace ───────────────────────────────────
def fig_iforest():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    norm = df[df["Is_Anomaly"] == 0]; ano = df[df["Is_Anomaly"] == 1]
    # jitter gear slightly for visibility
    jit = lambda n: (np.random.RandomState(42).rand(n) - 0.5) * 0.35
    ax.scatter(norm["Speed"], norm["nGear"] + jit(len(norm)), s=3, c="#9ecae1", alpha=0.4,
               label="Normal", rasterized=True)
    ax.scatter(ano["Speed"],  ano["nGear"]  + jit(len(ano)),  s=8, c=RED, alpha=0.75,
               label="Anomalous", rasterized=True)
    ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Gear")
    ax.set_title("Isolation Forest Anomalies in Speed--Gear Subspace", fontweight="bold")
    ax.legend(markerscale=3, frameon=True)
    fig.tight_layout(); fig.savefig(FIG / "isolation_forest_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  isolation_forest_results.png")


# ── 4. ROC curves ────────────────────────────────────────────────────────────────
def fig_roc():
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42).fit(Xtr, ytr)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for clf, name, col in [(rf, "Random Forest", BLUE), (lr, "Logistic Regression", GREEN)]:
        p = clf.predict_proba(Xte)[:, 1]; a = roc_auc_score(yte, p); fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase-1 Current-Mode ROC", fontweight="bold")
    ax.legend(loc="lower right", frameon=True); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  roc_curves.png")


# ── 5. Permutation importance (from saved authoritative CSV, with 95% CI) ─────────
def fig_permimp():
    pi = pd.read_csv(HERE / "stats_outputs" / "10_permutation_importance.csv")
    pi = pi[pi["Importance"] > 0].sort_values("Importance")   # keep the discriminative features
    err = np.vstack([pi["Importance"] - pi["CI_lo_95"], pi["CI_hi_95"] - pi["Importance"]])
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.barh(range(len(pi)), pi["Importance"], xerr=err, color=BLUE, edgecolor="black",
            lw=0.5, capsize=3, error_kw=dict(ecolor="#444444", lw=1))
    ax.set_yticks(range(len(pi))); ax.set_yticklabels(pi["Feature"], fontsize=10)
    ax.set_xlabel("Permutation importance ($\\Delta$AUC, 95% CI)")
    ax.set_title("Phase-1 Permutation Importance", fontweight="bold")
    for i, v in enumerate(pi["Importance"]):
        ax.text(min(v + 0.012, 0.188), i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, 0.21); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "stat_08_permutation_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  stat_08_permutation_importance.png (from saved CSV)")


def main():
    print(f"Writing Phase-1 figures -> {FIG}")
    fig_elbow(); fig_track_map(); fig_iforest(); fig_roc(); fig_permimp()
    print("Done. Regenerated 5 white-background figures.")


if __name__ == "__main__":
    main()
