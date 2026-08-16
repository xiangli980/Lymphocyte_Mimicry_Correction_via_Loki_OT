"""
Region-level tissue feature extractor.

Extracts patch-level feature embeddings from histopathology images
conditioned on tissue segmentation masks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


class RegionExtractor:
    """Extract region-level feature embeddings for tissue patches.

    Parameters
    ----------
    backbone:
        Name of the torchvision backbone to use (e.g. ``"resnet50"``).
    pretrained:
        Whether to load ImageNet-pretrained weights.
    device:
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    patch_size:
        Side length (in pixels) of square patches to extract per cell.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        device: str | None = None,
        patch_size: int = 64,
    ) -> None:
        self.patch_size = patch_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        encoder = getattr(models, backbone)(pretrained=pretrained)
        # Remove the classification head — keep up to the global average pool.
        self.feature_dim: int = encoder.fc.in_features
        self.encoder = nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def extract_from_patches(self, patches: list[Image.Image]) -> np.ndarray:
        """Return feature matrix of shape ``(N, feature_dim)`` for a list of PIL patches."""
        if not patches:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        tensors = torch.stack([_TRANSFORM(p) for p in patches]).to(self.device)
        feats = self.encoder(tensors).squeeze(-1).squeeze(-1)
        return feats.cpu().numpy().astype(np.float32)

    def extract_cell_patches(
        self,
        wsi_image: np.ndarray,
        cell_coords: np.ndarray,
    ) -> np.ndarray:
        """Crop square patches around each cell centre and extract features.

        Parameters
        ----------
        wsi_image:
            Whole-slide image as a ``(H, W, 3)`` uint8 NumPy array.
        cell_coords:
            Array of shape ``(N, 2)`` with ``(x, y)`` cell centre coordinates.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(N, feature_dim)``.
        """
        h, w = wsi_image.shape[:2]
        half = self.patch_size // 2
        patches: list[Image.Image] = []
        for x, y in cell_coords:
            x0 = int(max(0, x - half))
            y0 = int(max(0, y - half))
            x1 = int(min(w, x + half))
            y1 = int(min(h, y + half))
            crop = wsi_image[y0:y1, x0:x1]
            patches.append(Image.fromarray(crop))
        return self.extract_from_patches(patches)
