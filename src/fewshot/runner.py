"""Runs every (dataset, method, shots, draw) combination and appends one row per run to runs.csv.

Re-running skips rows that already exist, so an interrupted CPU run simply continues.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .config import Config
from .data import DatasetCache, draw_seed, few_shot_indices
from .methods import linear_probe, random_forest, scores

FIELDS = ["dataset", "method", "shots", "draw", "seed", "n_train", "n_test",
          "balanced_accuracy", "accuracy", "auc", "seconds"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device(name: str = "auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["dataset"], r["method"], int(r["shots"]), int(r["draw"])) for r in csv.DictReader(f)}


def append_row(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def handcrafted_cached(name: str, split: str, images: np.ndarray, feature_dir: Path) -> np.ndarray:
    from .handcrafted import handcrafted_features
    path = Path(feature_dir) / name / f"hog_rf_{split}.npy"
    if path.exists():
        return np.load(path)
    feats = handcrafted_features(images)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats


def run_dataset(cfg: Config, ds: DatasetCache):
    done = done_keys(cfg.results_csv)
    device = None
    holder: dict = {}
    for method in cfg.methods:
        todo = [(k, d) for k in cfg.protocol.shots for d in range(cfg.protocol.draws)
                if (ds.name, method, k, d) not in done]
        if not todo:
            log(f"{ds.name} / {method}: all {len(cfg.protocol.shots) * cfg.protocol.draws} runs already done")
            continue

        # ---- features shared by all draws of this method
        pool_f = test_f = None
        if method == "hog_rf":
            log(f"{ds.name} / {method}: handcrafted features")
            pool_f = handcrafted_cached(ds.name, "pool", ds.pool_images, cfg.feature_dir)
            test_f = handcrafted_cached(ds.name, "test", ds.test_images, cfg.feature_dir)
        elif method.endswith("_probe"):
            from .foundation import cached_features
            device = device or get_device(cfg.device)
            log(f"{ds.name} / {method}: extracting frozen features ({len(ds.pool_images)} pool + "
                f"{len(ds.test_images)} test images, cached after the first time)")
            pool_f = cached_features(method, ds.name, "pool", ds.pool_images, cfg.feature_dir, device,
                                     cfg.probe.batch_size, holder)
            test_f = cached_features(method, ds.name, "test", ds.test_images, cfg.feature_dir, device,
                                     cfg.probe.batch_size, holder)
        else:
            device = device or get_device(cfg.device)

        for shots, draw in todo:
            seed = draw_seed(cfg.protocol.base_seed, ds.name, shots, draw)
            idx = few_shot_indices(ds.pool_labels, shots, seed)   # identical for every method
            y_tr = ds.pool_labels[idx]
            t0 = time.time()
            if method == "hog_rf":
                probs = random_forest(pool_f[idx], y_tr, test_f, ds.n_classes, cfg.forest.n_estimators, seed % 2**31)
            elif method.endswith("_probe"):
                probs = linear_probe(pool_f[idx], y_tr, test_f, ds.n_classes, cfg.probe.C, cfg.probe.max_iter)
            elif method == "resnet18_finetune":
                from .finetune import finetune_predict
                probs = finetune_predict(ds.pool_images[idx], y_tr, ds.test_images, ds.n_classes,
                                         cfg.finetune, seed % 2**31, device)
            else:
                raise ValueError(method)
            s = scores(ds.test_labels, probs)
            row = {"dataset": ds.name, "method": method, "shots": shots, "draw": draw, "seed": seed,
                   "n_train": len(idx), "n_test": len(ds.test_labels), "seconds": round(time.time() - t0, 2), **s}
            append_row(cfg.results_csv, row)
            log(f"{ds.name} / {method} / {shots:>2}-shot / draw {draw}: "
                f"balanced acc {s['balanced_accuracy']:.3f}  AUC {s['auc']:.3f}  ({row['seconds']:.1f}s)")
        holder.clear()
