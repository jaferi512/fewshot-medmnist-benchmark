"""Frozen feature extractors: ImageNet ResNet-18, DINOv2, CLIP, MedSigLIP.

Each model sees the image resized (no cropping, so nothing at the border is lost) to its native input
size and normalised with its own mean/std. Features are computed once per dataset and cached.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
SIGLIP_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# method -> (input size, normalisation, Hugging Face id or None)
SPECS = {
    "resnet18_probe": (224, IMAGENET, None),
    "random_probe": (64, IMAGENET, None),
    "dinov2_probe": (224, IMAGENET, "facebook/dinov2-small"),
    "clip_probe": (224, CLIP_NORM, "openai/clip-vit-base-patch32"),
    "medsiglip_probe": (448, SIGLIP_NORM, "google/medsiglip-448"),
}


def preprocess(batch_u8: np.ndarray, size: int, norm, device) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> normalised (B, 3, size, size) float tensor."""
    x = torch.from_numpy(batch_u8).to(device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                          antialias=x.shape[-1] > size)
    mean = torch.tensor(norm[0], device=device)[None, :, None, None]
    std = torch.tensor(norm[1], device=device)[None, :, None, None]
    return (x - mean) / std


class Extractor:
    """Wraps a frozen backbone as `__call__(pixel_values) -> (B, D) features`."""

    def __init__(self, method: str, device):
        self.method = method
        self.size, self.norm, hf_id = SPECS[method]
        self.device = device
        if method in ("resnet18_probe", "random_probe"):
            from torchvision.models import ResNet18_Weights, resnet18
            torch.manual_seed(0)
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if method == "resnet18_probe" else None)
            m.fc = torch.nn.Identity()          # 512-d global-average-pooled features
            self.model = m
            self.fn = lambda px: self.model(px)
        elif method == "dinov2_probe":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(hf_id)
            self.fn = lambda px: self.model(pixel_values=px).pooler_output       # CLS token, 384-d
        elif method == "clip_probe":
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(hf_id)
            self.fn = lambda px: self.model(pixel_values=px).image_embeds        # 512-d
        elif method == "medsiglip_probe":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(hf_id)                       # gated: accept licence on HF first
            self.fn = lambda px: self.model.get_image_features(pixel_values=px)  # 1152-d
        else:
            raise ValueError(method)
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, images_u8: np.ndarray, batch_size: int = 64) -> np.ndarray:
        out = []
        for s in range(0, len(images_u8), batch_size):
            px = preprocess(images_u8[s:s + batch_size], self.size, self.norm, self.device)
            out.append(self.fn(px).float().cpu().numpy())
        return np.concatenate(out)


def cached_features(method: str, name: str, split: str, images: np.ndarray, feature_dir: Path, device,
                    batch_size: int, extractor_holder: dict) -> np.ndarray:
    """Load features from disk, or compute and save them. `extractor_holder` reuses a loaded model."""
    path = Path(feature_dir) / name / f"{method}_{split}.npy"
    if path.exists():
        return np.load(path)
    if method not in extractor_holder:
        extractor_holder.clear()          # keep only one large model in memory at a time
        extractor_holder[method] = Extractor(method, device)
    feats = extractor_holder[method](images, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats
