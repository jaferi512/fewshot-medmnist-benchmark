"""Draw the study-design figure: python scripts/make_method_figure.py  ->  results/figures/method_overview.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE, PANEL = "#0b0b0b", "#52514e", "#c9c8c3", "#f6f5f1"
BLUE, ORANGE, GREEN, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


def box(ax, x, y, w, h, title, lines, edge=LINE, face=PANEL, title_color=INK, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                linewidth=1.2, edgecolor=edge, facecolor=face))
    ax.text(x + 0.012, y + h - 0.03, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color=title_color)
    ax.text(x + 0.012, y + h - 0.085, "\n".join(lines), ha="left", va="top", fontsize=fs, color=MUTED, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=MUTED))


def main(out=Path("results/figures/method_overview.png")):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top, bottom = 0.855, 0.06

    ax.text(0.012, 0.975, "How few labels do you need?  Study design", fontsize=12, fontweight="bold", color=INK, va="top")

    # 1 data
    box(ax, 0.015, bottom, 0.165, top - bottom, "1  Four MedMNIST datasets", [
        "64×64 px, official splits", "",
        "Chest X-ray", "   PneumoniaMNIST · 2 classes", "",
        "Dermatoscopy", "   DermaMNIST · 7 classes", "",
        "Blood-cell microscopy", "   BloodMNIST · 8 classes", "",
        "Breast ultrasound", "   BreastMNIST · 2 classes"])

    # 2 pool, 3 draws
    box(ax, 0.235, 0.555, 0.17, top - 0.555, "2  Labelled pool", [
        "≤ 500 training images", "per class, fixed seed", "", "test set never touched"])
    box(ax, 0.235, bottom, 0.17, 0.43, "3  Few-shot draws", [
        "k = 1, 5, 10, 50 per class", "× 5 random draws", "",
        "the SAME images go", "to every method (paired)"])
    arrow(ax, 0.196, 0.70, 0.221, 0.70)            # data -> pool
    arrow(ax, 0.32, 0.540, 0.32, 0.505)            # pool -> draws

    # 4 methods
    mx, mw = 0.47, 0.28
    ax.text(mx - 0.012, top + 0.022, "4  Five classifiers, trained only on the k-shot images",
            fontsize=9.2, fontweight="bold", color=INK, va="bottom")
    rows = [  # (y, h, centre)
        (0.645, 0.21), (0.375, 0.235), (bottom, 0.28)]
    box(ax, mx, rows[0][0], mw, rows[0][1], "Classical", [
        "HOG + LBP + intensity histograms", "→ Random Forest (500 trees)"], edge=BLUE, face="#eef4fc", title_color=BLUE)
    box(ax, mx, rows[1][0], mw, rows[1][1], "Fine-tuned CNN", [
        "ResNet-18, ImageNet initialisation", "all weights trained · 200 AdamW steps",
        "128 px · rotation/scale/shift augmentation"], edge=ORANGE, face="#fdf1ec", title_color=ORANGE)
    box(ax, mx, rows[2][0], mw, rows[2][1], "Frozen backbone + linear probe", [
        "features computed once and cached,", "→ logistic regression (C = 1)", "",
        "ResNet-18 (ImageNet) · DINOv2 ViT-S/14", "CLIP ViT-B/32"], edge=GREEN, face="#edf8f3", title_color=GREEN)
    for y, h in rows:
        arrow(ax, 0.421, 0.275, mx - 0.016, y + h / 2)      # draws -> each method family
        arrow(ax, mx + mw + 0.016, y + h / 2, 0.795, y + h / 2 if y > 0.3 else 0.30)  # -> evaluation

    # 5 evaluation
    box(ax, 0.81, bottom, 0.178, top - bottom, "5  Evaluation", [
        "full official test set", "",
        "balanced accuracy (primary)", "macro one-vs-rest AUC", "",
        "mean ± 95% CI over", "the 5 draws (t-interval)", "",
        "paired differences", "between methods,", "average rank", "",
        "4 datasets × 5 methods", "× 4 budgets × 5 draws", "= 400 runs"])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
