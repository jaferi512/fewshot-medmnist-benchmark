"""Classical baseline: handcrafted texture/shape/intensity features (numpy + scikit-image)."""
from __future__ import annotations

import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize


def _grey(img_rgb: np.ndarray) -> np.ndarray:
    return (img_rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0


def handcrafted_features(images: np.ndarray, size: int = 64) -> np.ndarray:
    """Per image: HOG (shape/edges) + uniform LBP histogram (texture) + 32-bin histogram per colour channel.

    images: (N, H, W, 3) uint8. Returns (N, D) float32. Images are resized to `size` so the feature
    length does not depend on the MedMNIST resolution chosen.
    """
    feats = []
    for img in images:
        g = _grey(img)
        if g.shape[0] != size:
            g = resize(g, (size, size), anti_aliasing=True)
        h = hog(g, orientations=9, pixels_per_cell=(size // 8, size // 8), cells_per_block=(2, 2), feature_vector=True)
        lbp = local_binary_pattern((g * 255).astype(np.uint8), P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        col = np.concatenate([np.histogram(img[..., ch], bins=32, range=(0, 256), density=True)[0] * 256
                              for ch in range(3)])
        feats.append(np.concatenate([h, lbp_hist, col]).astype(np.float32))
    return np.stack(feats)
