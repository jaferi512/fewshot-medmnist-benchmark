"""MedMNIST loading, the labelled pool, and reproducible few-shot draws (numpy only)."""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetCache:
    name: str
    class_names: list[str]
    pool_images: np.ndarray    # (N, H, W, 3) uint8 - always RGB so every model sees the same input
    pool_labels: np.ndarray    # (N,)
    test_images: np.ndarray    # (M, H, W, 3) uint8
    test_labels: np.ndarray    # (M,)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def to_rgb(imgs: np.ndarray) -> np.ndarray:
    imgs = np.asarray(imgs)
    if imgs.ndim == 3:
        imgs = np.repeat(imgs[..., None], 3, axis=-1)
    return imgs.astype(np.uint8)


def load_medmnist_split(name: str, split: str, size: int, root: Path | None = None):
    """Download (first time) and load one split of a 2D MedMNIST dataset with the official package."""
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    root = Path(root or Path.home() / ".medmnist")
    root.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, info["python_class"])
    kwargs = {"split": split, "download": True, "root": str(root)}
    if size != 28:
        kwargs["size"] = size
    ds = cls(**kwargs)
    names = [info["label"][str(i)] for i in range(len(info["label"]))]
    return to_rgb(ds.imgs), np.asarray(ds.labels).reshape(-1).astype(int), names


def select_pool(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices of at most `per_class` random training images per class (sorted)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        idx.extend(rng.choice(members, size=min(per_class, len(members)), replace=False))
    return np.sort(np.array(idx, dtype=int))


def draw_seed(base_seed: int, dataset: str, shots: int, draw: int) -> int:
    """Deterministic seed per (dataset, shots, draw), stable across Python sessions and platforms."""
    return (base_seed * 1_000_003 + zlib.crc32(dataset.encode()) + shots * 7919 + draw * 104_729) % (2 ** 32)


def few_shot_indices(pool_labels: np.ndarray, shots: int, seed: int) -> np.ndarray:
    """`shots` pool indices per class (fewer if a class has fewer images). Same seed -> same draw for all methods."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(pool_labels):
        members = np.where(pool_labels == c)[0]
        idx.extend(rng.choice(members, size=min(shots, len(members)), replace=False))
    return np.array(idx, dtype=int)


def cache_path(cache_dir: Path, name: str, size: int) -> Path:
    return Path(cache_dir) / f"{name}_{size}.npz"


def save_cache(path: Path, c: DatasetCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pool_images=c.pool_images, pool_labels=c.pool_labels,
                        test_images=c.test_images, test_labels=c.test_labels,
                        class_names=np.array(json.dumps(c.class_names)))


def load_cache(path: Path, name: str) -> DatasetCache:
    d = np.load(path)
    return DatasetCache(name, json.loads(str(d["class_names"])), d["pool_images"], d["pool_labels"],
                        d["test_images"], d["test_labels"])


def prepare_medmnist(name: str, size: int, per_class: int, seed: int, cache_dir: Path) -> DatasetCache:
    path = cache_path(cache_dir, name, size)
    if path.exists():
        return load_cache(path, name)
    tr_x, tr_y, names = load_medmnist_split(name, "train", size)
    te_x, te_y, _ = load_medmnist_split(name, "test", size)
    pool = select_pool(tr_y, per_class, seed)
    c = DatasetCache(name, names, tr_x[pool], tr_y[pool], te_x, te_y)
    save_cache(path, c)
    return c


# --------------------------------------------------------------------------- toy data for smoke tests
def toy_dataset(name: str, size: int = 28, n_classes: int = 3, n_pool: int = 30, n_test: int = 20, seed: int = 0) -> DatasetCache:
    """Simple shapes, one family per class; 'toy_rgb' adds a class-dependent tint. Meaningless as data."""
    rng = np.random.default_rng(seed + (1 if name.endswith("rgb") else 0))
    yy, xx = np.mgrid[:size, :size] / (size - 1) - 0.5

    def make(n):
        imgs, labels = [], []
        for c in range(n_classes):
            for _ in range(n):
                cx, cy = rng.normal(0, 0.06, 2)
                r = 0.12 + 0.07 * c + rng.normal(0, 0.015)
                m = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2) if c % 2 == 0 else ((np.abs(xx - cx) < r) & (np.abs(yy - cy) < r))
                g = np.clip(0.2 + 0.6 * m + rng.normal(0, 0.08, m.shape), 0, 1)
                img = np.stack([g, g, g], -1)
                if name.endswith("rgb"):
                    img[..., c % 3] = np.clip(img[..., c % 3] * 1.2, 0, 1)
                imgs.append((img * 255).astype(np.uint8))
                labels.append(c)
        return np.stack(imgs), np.array(labels)

    px, py = make(n_pool)
    tx, ty = make(n_test)
    return DatasetCache(name, [f"class_{c}" for c in range(n_classes)], px, py, tx, ty)
