"""ResNet-18 fine-tuned end to end on the few labelled images (the 'usual way', as in the TB study)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FinetuneConfig
from .foundation import IMAGENET, preprocess


def random_affine(x: torch.Tensor, gen: torch.Generator, rotation_deg: float, scale: float, shift: float) -> torch.Tensor:
    """Per-image random rotation, isotropic scaling and translation (no flips). x: (B, C, H, W) in [0, 1]."""
    b = x.shape[0]
    ang = (torch.rand(b, generator=gen) * 2 - 1) * rotation_deg * math.pi / 180
    s = 1 + (torch.rand(b, generator=gen) * 2 - 1) * scale
    t = (torch.rand(b, 2, generator=gen) * 2 - 1) * shift * 2          # grid coordinates span [-1, 1]
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta.to(x), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def build_resnet18(n_classes: int, pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def finetune_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, n_classes: int,
                     cfg: FinetuneConfig, seed: int, device) -> np.ndarray:
    """Train on (train_x, train_y), return class probabilities for test_x. Images are (N, H, W, 3) uint8."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_resnet18(n_classes, cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    # [0,1] tensors at the training resolution; normalisation is applied after augmentation
    x01 = preprocess(train_x, cfg.input_size, ((0, 0, 0), (1, 1, 1)), "cpu")
    y = torch.from_numpy(train_y.astype(np.int64))
    mean = torch.tensor(IMAGENET[0])[None, :, None, None]
    std = torch.tensor(IMAGENET[1])[None, :, None, None]

    model.train()
    for _ in range(cfg.steps):
        idx = torch.randint(0, len(y), (cfg.batch_size,), generator=gen)   # with replacement: works for 1-shot
        xb = random_affine(x01[idx], gen, cfg.rotation_deg, cfg.scale, cfg.shift)
        xb = ((xb - mean) / std).to(device)
        loss = F.cross_entropy(model(xb), y[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(test_x), 256):
            xb = preprocess(test_x[s:s + 256], cfg.input_size, IMAGENET, device)
            probs.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(probs)
