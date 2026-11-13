import csv
import json

import numpy as np
import pytest

from fewshot.__main__ import load_datasets
from fewshot.analyze import run as analyze
from fewshot.config import smoke_config
from fewshot.runner import log, run_dataset


def test_end_to_end_classical_baseline_and_resume(tmp_path):
    """Real pipeline on toy data with the torch-free method: draws -> features -> RF -> CSV -> resume -> report."""
    cfg = smoke_config(tmp_path)
    cfg.methods = ("hog_rf",)
    datasets = load_datasets(cfg, log)
    for ds in datasets:
        run_dataset(cfg, ds)
    rows = list(csv.DictReader(open(cfg.results_csv, encoding="utf-8")))
    assert len(rows) == 2 * 2 * 2                                  # datasets x shots x draws
    assert {int(r["n_train"]) for r in rows} == {3, 15}            # 3 classes x {1, 5} shots
    assert np.mean([float(r["balanced_accuracy"]) for r in rows if r["shots"] == "5"]) > 0.6

    for ds in datasets:                                            # resume: nothing new is added
        run_dataset(cfg, ds)
    assert len(list(csv.DictReader(open(cfg.results_csv, encoding="utf-8")))) == len(rows)

    meta = json.loads((cfg.out_dir / "datasets.json").read_text(encoding="utf-8"))
    report = analyze(cfg, {d: v["n_classes"] for d, v in meta.items()}).read_text(encoding="utf-8")
    assert "Balanced accuracy" in report and "HOG + RF" in report
    figs = sorted(p.name for p in (cfg.results_dir / "figures").glob("*.png"))
    assert figs == ["fewshot_auc.png", "fewshot_balanced_accuracy.png", "fewshot_mean_over_datasets.png"]


def test_report_with_several_methods(tmp_path):
    """Analysis with synthetic rows for all default methods (checks tables, ranking and paired comparison)."""
    from fewshot.config import DEFAULT_METHODS
    from fewshot.runner import FIELDS
    cfg = smoke_config(tmp_path)
    cfg.methods = DEFAULT_METHODS
    rng = np.random.default_rng(0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for d in ("pneumoniamnist", "bloodmnist"):
            for mi, m in enumerate(DEFAULT_METHODS):
                for k in (1, 5, 10, 50):
                    for dr in range(5):
                        v = min(0.99, 0.4 + 0.08 * mi + 0.1 * np.log10(k) + rng.normal(0, 0.02))
                        w.writerow({"dataset": d, "method": m, "shots": k, "draw": dr, "seed": 0, "n_train": k,
                                    "n_test": 100, "balanced_accuracy": v, "accuracy": v, "auc": v, "seconds": 1})
    report = analyze(cfg, {"pneumoniamnist": 2, "bloodmnist": 8}).read_text(encoding="utf-8")
    assert "Paired comparison" in report and "Overall ranking" in report
    assert report.index("CLIP frozen") > 0


def test_finetune_and_frozen_extractor_shapes():
    torch = pytest.importorskip("torch")
    from fewshot.config import FinetuneConfig
    from fewshot.data import toy_dataset
    from fewshot.finetune import finetune_predict, random_affine
    from fewshot.foundation import Extractor

    x = torch.rand(2, 3, 16, 16)
    g = torch.Generator().manual_seed(0)
    assert torch.allclose(random_affine(x, g, 0, 0, 0), x, atol=1e-5)    # zero transform = identity

    ds = toy_dataset("toy_rgb", 28, n_pool=4, n_test=3)
    cfg = FinetuneConfig(steps=2, input_size=32, pretrained=False, batch_size=4)
    p = finetune_predict(ds.pool_images, ds.pool_labels, ds.test_images, 3, cfg, 0, "cpu")
    assert p.shape == (9, 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    f = Extractor("random_probe", "cpu")(ds.test_images, batch_size=4)
    assert f.shape == (9, 512)
