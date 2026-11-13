"""Command line.

    python -m fewshot prepare    # download MedMNIST and build the labelled pools (once)
    python -m fewshot run        # all methods x label budgets x draws -> outputs/runs.csv (resumable)
    python -m fewshot analyze    # tables + figures -> outputs/results/
    python -m fewshot all        # the three steps above
    python -m fewshot smoke      # 1-minute end-to-end check on toy data, no downloads

Useful options:
    --datasets pneumoniamnist bloodmnist    run a subset
    --methods hog_rf dinov2_probe           run a subset of methods (add medsiglip_probe if you have access)
    --shots 1 5 10 50 --draws 5             the protocol
    --device cpu --threads 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATASETS, DEFAULT_METHODS, METHODS, Config, smoke_config
from .data import prepare_medmnist, toy_dataset


def load_datasets(cfg: Config, log):
    out = []
    for name in cfg.data.datasets:
        if cfg.synthetic:
            ds = toy_dataset(name, cfg.data.size, n_pool=cfg.data.pool_per_class)
        else:
            log(f"{name}: loading (downloads {cfg.data.size}x{cfg.data.size} MedMNIST the first time)")
            ds = prepare_medmnist(name, cfg.data.size, cfg.data.pool_per_class, cfg.data.pool_seed, cfg.cache_dir)
        log(f"{name}: {ds.n_classes} classes, pool {len(ds.pool_labels)} labelled candidates, "
            f"test {len(ds.test_labels)} images")
        out.append(ds)
    meta = {ds.name: {"n_classes": ds.n_classes, "class_names": ds.class_names,
                      "pool": int(len(ds.pool_labels)), "test": int(len(ds.test_labels))} for ds in out}
    (cfg.out_dir / "datasets.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fewshot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "analyze", "all", "smoke"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(METHODS))
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10, 50])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)

    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    if a.command == "smoke":
        cfg = smoke_config(a.out_dir / "smoke")
    else:
        cfg = Config(out_dir=a.out_dir, methods=tuple(a.methods))
        cfg.data.datasets, cfg.data.size = tuple(a.datasets), a.size
        cfg.protocol.shots, cfg.protocol.draws = tuple(sorted(a.shots)), a.draws
    cfg.device = a.device
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    from .runner import log, run_dataset
    if a.command in ("prepare", "run", "all", "smoke"):
        datasets = load_datasets(cfg, log)
    if a.command in ("run", "all", "smoke"):
        n_runs = len(datasets) * len(cfg.methods) * len(cfg.protocol.shots) * cfg.protocol.draws
        log(f"{n_runs} runs in total ({len(cfg.methods)} methods, shots {list(cfg.protocol.shots)}, "
            f"{cfg.protocol.draws} draws); finished runs are skipped")
        for ds in datasets:
            run_dataset(cfg, ds)
    if a.command in ("analyze", "all", "smoke"):
        from .analyze import run as analyze
        meta = json.loads((cfg.out_dir / "datasets.json").read_text(encoding="utf-8"))
        path = analyze(cfg, {d: v["n_classes"] for d, v in meta.items()})
        log(f"Wrote {path} and figures in {cfg.results_dir / 'figures'}")
    if a.command == "smoke":
        log("Smoke test finished - every stage runs end to end (toy data, numbers are meaningless).")


if __name__ == "__main__":
    main()
