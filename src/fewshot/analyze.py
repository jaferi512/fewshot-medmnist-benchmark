"""Summary statistics, tables and figures from runs.csv (numpy/scipy/matplotlib only)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .config import METHODS, Config

# Categorical palette slots 1-6 (validated for colour-vision deficiency); markers give a second encoding.
STYLE = {
    "hog_rf":            ("#2a78d6", "s"),
    "resnet18_finetune": ("#eb6834", "o"),
    "resnet18_probe":    ("#1baf7a", "^"),
    "dinov2_probe":      ("#eda100", "D"),
    "clip_probe":        ("#e87ba4", "v"),
    "medsiglip_probe":   ("#008300", "P"),
    "random_probe":      ("#8a8984", "x"),
}
SHORT = {
    "hog_rf": "HOG + RF", "resnet18_finetune": "ResNet-18 fine-tuned", "resnet18_probe": "ResNet-18 frozen",
    "dinov2_probe": "DINOv2 frozen", "clip_probe": "CLIP frozen", "medsiglip_probe": "MedSigLIP frozen",
    "random_probe": "Random CNN frozen",
}
PRETTY = {"pneumoniamnist": "PneumoniaMNIST (chest X-ray)", "dermamnist": "DermaMNIST (skin)",
          "bloodmnist": "BloodMNIST (blood cells)", "breastmnist": "BreastMNIST (breast ultrasound)",
          "octmnist": "OCTMNIST (retinal OCT)"}
INK, MUTED = "#0b0b0b", "#52514e"


def read_runs(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("shots", "draw", "n_train", "n_test"):
            r[k] = int(r[k])
        for k in ("balanced_accuracy", "accuracy", "auc", "seconds"):
            r[k] = float(r[k])
    return rows


def mean_ci(values, alpha: float = 0.05):
    """Mean and half-width of the t-based 95% CI across random draws."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return float(v[0]), float("nan"), 1
    half = stats.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half), len(v)


def summarise(rows: list[dict], metric: str = "balanced_accuracy") -> dict:
    """{dataset: {method: {shots: (mean, ci_half, n)}}}"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"], r["shots"])].append(r[metric])
    out = defaultdict(lambda: defaultdict(dict))
    for (d, m, k), vals in groups.items():
        out[d][m][k] = mean_ci(vals)
    return out


def paired_difference(rows: list[dict], method_a: str, method_b: str, metric: str = "balanced_accuracy") -> dict:
    """Mean (a - b) over draws, per dataset and shots, with t-based 95% CI. Draws are paired by design."""
    by = {(r["dataset"], r["method"], r["shots"], r["draw"]): r[metric] for r in rows}
    out = defaultdict(dict)
    for (d, m, k, dr), v in by.items():
        if m != method_a or (d, method_b, k, dr) not in by:
            continue
        out[(d, k)].setdefault("diffs", []).append(v - by[(d, method_b, k, dr)])
    return {key: mean_ci(val["diffs"]) for key, val in out.items()}


def average_rank(summary: dict, methods: list[str]) -> dict:
    """Mean rank of each method across all (dataset, shots) settings (1 = best)."""
    ranks = defaultdict(list)
    for d, per_m in summary.items():
        shots = sorted({k for m in per_m for k in per_m[m]})
        for k in shots:
            vals = [(per_m[m][k][0], m) for m in methods if m in per_m and k in per_m[m]]
            if len(vals) < 2:
                continue
            order = stats.rankdata([-v for v, _ in vals])
            for (_, m), r in zip(vals, order):
                ranks[m].append(r)
    return {m: float(np.mean(r)) for m, r in ranks.items()}


# --------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt


def plot_curves(summary: dict, methods: list[str], n_classes: dict, path: Path, metric_label: str):
    plt = _plt()
    datasets = list(summary)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.5), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for m in methods:
            if m not in summary[d]:
                continue
            ks = sorted(summary[d][m])
            mu = np.array([summary[d][m][k][0] for k in ks])
            ci = np.nan_to_num(np.array([summary[d][m][k][1] for k in ks]))
            color, marker = STYLE.get(m, ("#52514e", "o"))
            ax.plot(ks, mu, color=color, marker=marker, markersize=6, label=SHORT.get(m, m),
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.fill_between(ks, mu - ci, mu + ci, color=color, alpha=0.12, linewidth=0)
        ks_all = sorted({k for m in summary[d] for k in summary[d][m]})
        if d in n_classes:
            ax.axhline(1 / n_classes[d], color=MUTED, ls=":", lw=1)
            ax.text(ks_all[0], 1 / n_classes[d], " chance", va="bottom", ha="left", fontsize=7, color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks(ks_all, [str(k) for k in ks_all])
        ax.minorticks_off()
        ax.set_xlabel("Labelled images per class")
        ax.set_title(PRETTY.get(d, d))
    axes[0, 0].set_ylabel(metric_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mean_over_datasets(summary: dict, methods: list[str], path: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    shots = sorted({k for d in summary for m in summary[d] for k in summary[d][m]})
    for m in methods:
        mu = []
        for k in shots:
            vals = [summary[d][m][k][0] for d in summary if m in summary[d] and k in summary[d][m]]
            mu.append(np.mean(vals) if len(vals) == len(summary) else np.nan)
        if np.all(np.isnan(mu)):
            continue
        color, marker = STYLE.get(m, ("#52514e", "o"))
        ax.plot(shots, mu, color=color, marker=marker, markersize=6, label=SHORT.get(m, m),
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log"); ax.set_xticks(shots, [str(k) for k in shots]); ax.minorticks_off()
    ax.set_xlabel("Labelled images per class"); ax.set_ylabel("Balanced accuracy (mean over datasets)")
    ax.set_title("Average over datasets"); ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- report
def _cell(t):
    mu, half, n = t
    if np.isnan(mu):
        return "–"
    return f"{mu:.3f} ± {half:.3f}" if np.isfinite(half) else f"{mu:.3f}"


def markdown(rows, summary_bacc, summary_auc, methods, n_classes, cfg_dict) -> str:
    shots = sorted({r["shots"] for r in rows})
    draws = sorted({r["draw"] for r in rows})
    L = ["# Results", "",
         f"Mean ± 95% CI half-width over **{len(draws)} random draws** of the labelled images "
         f"(the same draws for every method). Metric: **balanced accuracy** on the full official test set "
         f"(chance = 1 / number of classes). Configuration: `outputs/config.json`.", ""]
    ranks = average_rank(summary_bacc, methods)
    if ranks:
        L += ["## Overall ranking", "", "Average rank across all dataset × label-budget settings (1 = best):", "",
              "| Method | Average rank |", "|---|---|"]
        for m, r in sorted(ranks.items(), key=lambda x: x[1]):
            L.append(f"| {METHODS.get(m, m)} | {r:.2f} |")
        L.append("")
    for metric_name, summary in (("Balanced accuracy", summary_bacc), ("Macro AUC (one-vs-rest)", summary_auc)):
        L += [f"## {metric_name}", ""]
        for d in summary:
            L += [f"### {PRETTY.get(d, d)}" + (f" — {n_classes[d]} classes" if d in n_classes else ""), "",
                  "| Method | " + " | ".join(f"{k}-shot" for k in shots) + " |",
                  "|---|" + "---|" * len(shots)]
            for m in methods:
                if m in summary[d]:
                    L.append(f"| {SHORT.get(m, m)} | " + " | ".join(_cell(summary[d][m].get(k, (np.nan, np.nan, 0)))
                                                                for k in shots) + " |")
            L.append("")
    probes = [m for m in methods if m.endswith("_probe") and m != "random_probe"]
    if "resnet18_finetune" in methods and probes:
        best = min(probes, key=lambda m: average_rank(summary_bacc, probes).get(m, 99))
        diff = paired_difference(rows, best, "resnet18_finetune")
        if diff:
            L += [f"## Paired comparison: {SHORT[best]} − ResNet-18 fine-tuned", "",
                  "Difference in balanced accuracy, paired by draw (positive = frozen features better).", "",
                  "| Dataset | " + " | ".join(f"{k}-shot" for k in shots) + " |", "|---|" + "---|" * len(shots)]
            for d in summary_bacc:
                L.append(f"| {PRETTY.get(d, d)} | " + " | ".join(
                    (lambda t: f"{t[0]:+.3f} ± {t[1]:.3f}" if np.isfinite(t[1]) else "–")(diff.get((d, k), (np.nan, np.nan, 0)))
                    for k in shots) + " |")
            L.append("")
    secs = defaultdict(float)
    for r in rows:
        secs[r["method"]] += r["seconds"]
    L += ["## Compute (training + prediction, excluding one-off feature extraction)", "",
          "| Method | Total minutes |", "|---|---|"] + [f"| {SHORT.get(m, m)} | {secs[m] / 60:.1f} |" for m in methods if m in secs]
    return "\n".join(L) + "\n"


def run(cfg: Config, n_classes: dict):
    rows = read_runs(cfg.results_csv)
    methods = [m for m in cfg.methods if any(r["method"] == m for r in rows)]
    s_bacc = summarise(rows, "balanced_accuracy")
    s_auc = summarise(rows, "auc")
    out = cfg.results_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plot_curves(s_bacc, methods, n_classes, out / "figures" / "fewshot_balanced_accuracy.png", "Balanced accuracy")
    plot_curves(s_auc, methods, {}, out / "figures" / "fewshot_auc.png", "Macro AUC")
    plot_mean_over_datasets(s_bacc, methods, out / "figures" / "fewshot_mean_over_datasets.png")
    (out / "RESULTS.md").write_text(markdown(rows, s_bacc, s_auc, methods, n_classes, cfg.to_dict()), encoding="utf-8")
    plain = {d: {m: {str(k): v for k, v in per_k.items()} for m, per_k in per_m.items()} for d, per_m in s_bacc.items()}
    (out / "summary_balanced_accuracy.json").write_text(json.dumps(plain, indent=2), encoding="utf-8")
    return out / "RESULTS.md"
