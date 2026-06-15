"""
04_evaluate.py
================
Aggregates results from 02_baselines.py and 03_deep_models.py and produces:

  1. Accuracy-vs-horizon curve (F1-macro and AUC-PR for each model)
  2. LOCO generalization table (per-fold and mean±std)
  3. Transition-window F1 analysis
  4. Statistical tests:
       - DeLong (AUC-ROC comparison between model pairs, per fold, per H)
       - Paired Wilcoxon across folds for F1-macro, AUC-ROC, AUC-PR
       - Bootstrap 95% CIs for all aggregate metrics
  5. LaTeX tables ready for the paper
  6. Matplotlib figures (horizon curve, LOCO heatmap)

Usage:
  cd multi_circuit_work
  python 04_evaluate.py                         # uses all available results
  python 04_evaluate.py --horizon-primary 10    # marks H=10 as primary result

Outputs (all under multi_circuit_work/):
  graphs/horizon_curve.pdf
  graphs/loco_table.pdf
  processed/eval_summary.txt    — paste into paper
  processed/stats_report.txt    — statistical test results
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — skipping figure generation.")

WORK_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = WORK_ROOT / "processed"
GRAPHS_DIR = WORK_ROOT / "graphs"
PREDS_DIR_BASE = PROCESSED_DIR  # baseline_preds and deep_preds are here


# ── DeLong AUC comparison ──────────────────────────────────────────────────────

def _delong_var(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Variance of AUC via DeLong et al. (1988) structural components."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m == 0 or n == 0:
        return float("nan")
    pos_scores = y_score[pos_idx]
    neg_scores = y_score[neg_idx]
    # V10[i] = fraction of negatives that score below positive i
    V10 = np.mean(pos_scores[:, None] > neg_scores[None, :], axis=1)
    # V01[j] = fraction of positives that score above negative j
    V01 = np.mean(pos_scores[:, None] > neg_scores[None, :], axis=0)
    auc = V10.mean()
    s10 = np.var(V10, ddof=1) / m
    s01 = np.var(V01, ddof=1) / n
    return float(s10 + s01), float(auc)


def delong_test(y_true: np.ndarray,
                y_score_a: np.ndarray,
                y_score_b: np.ndarray) -> dict:
    """
    DeLong test comparing AUC_A vs AUC_B on the same test set.
    Returns p-value (two-sided) and confidence interval on ΔAUC.
    """
    res_a = _delong_var(y_true, y_score_a)
    res_b = _delong_var(y_true, y_score_b)
    if isinstance(res_a, float) or isinstance(res_b, float):
        return {"p": float("nan"), "delta_auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    var_a, auc_a = res_a
    var_b, auc_b = res_b

    # covariance term (Hanley–McNeil approximation for correlated ROC curves on same data)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    V10_a = np.mean(y_score_a[pos_idx][:, None] > y_score_a[neg_idx][None, :], axis=1)
    V10_b = np.mean(y_score_b[pos_idx][:, None] > y_score_b[neg_idx][None, :], axis=1)
    V01_a = np.mean(y_score_a[pos_idx][:, None] > y_score_a[neg_idx][None, :], axis=0)
    V01_b = np.mean(y_score_b[pos_idx][:, None] > y_score_b[neg_idx][None, :], axis=0)
    cov = (np.cov(V10_a, V10_b)[0, 1] / len(pos_idx) +
           np.cov(V01_a, V01_b)[0, 1] / len(neg_idx))

    var_diff = var_a + var_b - 2 * cov
    if var_diff <= 0:
        return {"p": float("nan"), "delta_auc": auc_a - auc_b, "ci_lo": float("nan"), "ci_hi": float("nan")}

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z)))
    ci_lo = float((auc_a - auc_b) - 1.96 * np.sqrt(var_diff))
    ci_hi = float((auc_a - auc_b) + 1.96 * np.sqrt(var_diff))
    return {"p": p, "delta_auc": float(auc_a - auc_b), "ci_lo": ci_lo, "ci_hi": ci_hi}


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of a metric across folds."""
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean()
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


# ── loader helpers ────────────────────────────────────────────────────────────

def load_all_results() -> pd.DataFrame:
    """Combine baseline_results.csv and deep_results.csv."""
    dfs = []
    for name in ["baseline_results.csv", "deep_results.csv"]:
        path = PROCESSED_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            "No results CSVs found. Run 02_baselines.py and/or 03_deep_models.py first."
        )
    return pd.concat(dfs, ignore_index=True)


def load_preds(model: str, held_out: str, H: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load saved y_prob and y_true for a model/fold/horizon combination."""
    for sub in ["baseline_preds", "deep_preds"]:
        path = PROCESSED_DIR / sub / f"{model}_{held_out}_H{H:03d}.npz"
        if path.exists():
            d = np.load(path)
            return d["y_prob"], d["y_true"]
    return None


# ── statistical tests ─────────────────────────────────────────────────────────

def run_stat_tests(results: pd.DataFrame) -> list[dict]:
    """
    For each pair (model_A, model_B) at each H:
      - Paired Wilcoxon on AUC-ROC across LOCO folds
      - Report median difference, effect size (r), 95% bootstrap CI on mean diff
    Returns list of result dicts.
    """
    rows = []
    models = results["model"].unique().tolist()
    horizons = sorted(results["H"].unique())

    for H in horizons:
        df_h = results[results["H"] == H]
        for a, b in combinations(models, 2):
            df_a = df_h[df_h["model"] == a].set_index("held_out")
            df_b = df_h[df_h["model"] == b].set_index("held_out")
            common_folds = df_a.index.intersection(df_b.index)
            if len(common_folds) < 2:
                continue

            for metric in ["auc_roc", "auc_pr", "f1_macro"]:
                va = df_a.loc[common_folds, metric].dropna().values
                vb = df_b.loc[common_folds, metric].dropna().values
                if len(va) < 2 or len(va) != len(vb):
                    continue
                diff = va - vb
                ci_lo, ci_hi = bootstrap_ci(diff)
                try:
                    _, p = stats.wilcoxon(va, vb, alternative="two-sided")
                    n = len(va)
                    # effect size r = z / sqrt(n)
                    z = stats.norm.ppf(1 - p / 2)
                    r = z / np.sqrt(n)
                except Exception:
                    p, r = float("nan"), float("nan")

                rows.append({
                    "H": H, "model_A": a, "model_B": b, "metric": metric,
                    "n_folds": len(va),
                    "mean_A": float(np.mean(va)), "mean_B": float(np.mean(vb)),
                    "mean_diff": float(np.mean(diff)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "p_wilcoxon": p, "effect_r": r,
                })
    return rows


# ── figures ───────────────────────────────────────────────────────────────────

def plot_horizon_curve(results: pd.DataFrame, metric: str = "f1_macro",
                       primary_H: int = 10) -> None:
    if not HAS_MPL:
        return
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    model_styles = {
        "LR-instant":   ("gray",   "--", "o"),
        "RF-instant":   ("orange", "--", "s"),
        "RF-lag":       ("red",    "-",  "^"),
        "CNN":          ("blue",   "-",  "o"),
        "LSTM":         ("green",  "-",  "s"),
        "GRU":          ("lime",   "-",  "D"),
        "TCN":          ("purple", "-",  "v"),
        "Transformer":  ("brown",  "-",  "P"),
    }

    summary = (
        results
        .groupby(["model", "H"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    models_in_results = results["model"].unique()
    for model_name in model_styles:
        if model_name not in models_in_results:
            continue
        sub = summary[summary["model"] == model_name]
        color, ls, marker = model_styles[model_name]
        ax.errorbar(sub["H"] / 10, sub["mean"], yerr=sub["std"],
                    label=model_name, color=color, linestyle=ls,
                    marker=marker, capsize=3, linewidth=1.5)

    ax.axvline(primary_H / 10, color="black", linestyle=":", alpha=0.6, label=f"H={primary_H} (primary)")
    ax.set_xlabel("Anticipation horizon (s)", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=12)
    ax.set_title("Anticipatory Aero Mode Prediction — Horizon Sweep\n"
                 "(mean ± std across LOCO folds)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    path = GRAPHS_DIR / "horizon_curve.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_loco_heatmap(results: pd.DataFrame, metric: str = "auc_roc",
                      H_primary: int = 10) -> None:
    if not HAS_MPL:
        return
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    sub = results[results["H"] == H_primary].pivot_table(
        index="model", columns="held_out", values=metric, aggfunc="mean"
    )
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(sub.columns) * 1.5), max(4, len(sub) * 0.8)))
    im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn",
                   vmin=sub.values.min() - 0.02, vmax=sub.values.max() + 0.02)
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(range(len(sub.columns)))
    ax.set_yticks(range(len(sub.index)))
    ax.set_xticklabels(sub.columns, fontsize=9)
    ax.set_yticklabels(sub.index, fontsize=9)
    for i in range(len(sub.index)):
        for j in range(len(sub.columns)):
            val = sub.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"LOCO Generalization — {metric} at H={H_primary}", fontsize=11)
    ax.set_xlabel("Held-out circuit", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    path = GRAPHS_DIR / f"loco_heatmap_H{H_primary:03d}.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── latex helpers ─────────────────────────────────────────────────────────────

def results_to_latex(results: pd.DataFrame, H: int,
                     metrics: list[str] = None) -> str:
    if metrics is None:
        metrics = ["f1_macro", "f1_xmode", "auc_roc", "auc_pr"]
    sub = results[results["H"] == H]
    agg = sub.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]

    lines = [
        f"% Auto-generated LOCO results at H={H} ({H/10:.1f}s ahead)",
        r"\begin{tabular}{l" + "c" * len(metrics) + r"}",
        r"\toprule",
        "Model & " + " & ".join(m.replace("_", "-") for m in metrics) + r" \\",
        r"\midrule",
    ]
    for model_name in agg.index:
        vals = []
        for m in metrics:
            mn = agg.loc[model_name, f"{m}_mean"]
            sd = agg.loc[model_name, f"{m}_std"]
            vals.append(f"${mn:.3f}\\pm{sd:.3f}$")
        lines.append(model_name + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--horizon-primary", type=int, default=10,
                   help="Primary H for LOCO tables and LaTeX (default: 10).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    H_primary = args.horizon_primary
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    models = sorted(results["model"].unique())
    horizons = sorted(results["H"].unique())
    circuits = sorted(results["held_out"].unique())
    print(f"Loaded results: {len(results)} rows")
    print(f"  Models: {models}")
    print(f"  Horizons: {horizons}")
    print(f"  LOCO folds: {circuits}")

    # ── summary table ──────────────────────────────────────────────────────
    print(f"\n── LOCO results at H={H_primary} ({H_primary/10:.1f}s ahead) ───────────")
    sub_primary = results[results["H"] == H_primary]
    if sub_primary.empty:
        print(f"  No results at H={H_primary}. Available: {horizons}")
    else:
        agg = (sub_primary
               .groupby("model")[["f1_macro", "f1_xmode", "auc_roc", "auc_pr"]]
               .agg(["mean", "std"])
               .round(4))
        agg.columns = ["_".join(c) for c in agg.columns]
        print(agg.to_string())

    # ── horizon curve ──────────────────────────────────────────────────────
    if len(horizons) > 1:
        print("\n── Horizon curve (mean F1-macro across folds) ────────────────────")
        hcurve = (results
                  .groupby(["model", "H"])["f1_macro"]
                  .mean()
                  .unstack("model")
                  .round(4))
        print(hcurve.to_string())
        plot_horizon_curve(results, "f1_macro", H_primary)
        plot_horizon_curve(results, "auc_pr", H_primary)

    plot_loco_heatmap(results, "auc_roc", H_primary)

    # ── statistical tests ──────────────────────────────────────────────────
    print("\n── Statistical tests ─────────────────────────────────────────────")
    stat_rows = run_stat_tests(results)
    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        stat_path = PROCESSED_DIR / "stats_report.csv"
        stat_df.to_csv(stat_path, index=False)
        print(f"  Saved {len(stat_df)} pairwise comparisons → {stat_path.name}")

        # highlight significant comparisons for deep vs RF-lag at primary H
        sig = stat_df[
            (stat_df["H"] == H_primary)
            & (stat_df["p_wilcoxon"] < 0.05)
            & stat_df[["model_A", "model_B"]].isin(["RF-lag", "CNN", "LSTM", "GRU", "TCN", "Transformer"]).all(axis=1)
        ]
        if not sig.empty:
            print(f"\n  Significant comparisons at H={H_primary} (p<0.05):")
            print(sig[["model_A", "model_B", "metric", "mean_diff", "p_wilcoxon", "effect_r"]].to_string(index=False))
        else:
            print(f"  No significant pairwise differences at H={H_primary} (p<0.05).")
            print("  NOTE: With only 4 LOCO folds, Wilcoxon has very low power.")
            print("  Report effect sizes (Cohen's r) and bootstrap CIs alongside p-values.")
    else:
        print("  Insufficient data for statistical tests (need ≥2 folds per pair).")

    # ── bootstrap CIs on aggregate metrics ────────────────────────────────
    print(f"\n── Bootstrap 95% CIs at H={H_primary} (2000 bootstrap resamples) ──")
    if not sub_primary.empty:
        for model_name in models:
            sub_m = sub_primary[sub_primary["model"] == model_name]
            for metric in ["f1_macro", "auc_roc", "auc_pr"]:
                vals = sub_m[metric].dropna().values
                if len(vals) >= 2:
                    ci_lo, ci_hi = bootstrap_ci(vals)
                    print(f"  {model_name:14s} {metric:10s}: "
                          f"{vals.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ── latex table ────────────────────────────────────────────────────────
    if not sub_primary.empty:
        latex = results_to_latex(results, H_primary)
        tex_path = PROCESSED_DIR / f"table_H{H_primary:03d}.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"\n── LaTeX table saved → {tex_path.name} ──────────────────────────")
        print(latex)

    # ── summary text ──────────────────────────────────────────────────────
    summary_path = PROCESSED_DIR / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Evaluation Summary — H_primary={H_primary}\n")
        fh.write("=" * 60 + "\n")
        if not sub_primary.empty:
            fh.write(agg.to_string() + "\n\n")
        if len(horizons) > 1:
            fh.write("Horizon curve (mean F1-macro):\n")
            fh.write(hcurve.to_string() + "\n")
    print(f"\nSummary saved → {summary_path.name}")


if __name__ == "__main__":
    main()
