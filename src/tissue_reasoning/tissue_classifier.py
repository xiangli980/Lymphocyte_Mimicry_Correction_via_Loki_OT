"""
Tissue type classifier.

Assigns each detected cell to a tissue region class (e.g. tumour stroma,
lymphoid aggregate, necrosis) based on the tissue segmentation mask and
the surrounding context.
"""

from __future__ import annotations

import numpy as np


class TissueClassifier:
    """Map cell coordinates to tissue-type labels using a segmentation mask.

    Parameters
    ----------
    num_classes:
        Number of distinct tissue classes encoded in the mask.
    background_class:
        Integer value that represents background / unlabelled pixels.
    """

    def __init__(self, num_classes: int = 5, background_class: int = 0) -> None:
        self.num_classes = num_classes
        self.background_class = background_class

    def assign_tissue_labels(
        self,
        tissue_mask: np.ndarray,
        cell_coords: np.ndarray,
    ) -> np.ndarray:
        """Return the tissue-type label for each cell.

        Parameters
        ----------
        tissue_mask:
            Integer array of shape ``(H, W)`` where each pixel value is a
            tissue class index in ``[0, num_classes)``.
        cell_coords:
            Array of shape ``(N, 2)`` with ``(x, y)`` cell-centre pixel
            coordinates (column-first / OpenCV convention).

        Returns
        -------
        np.ndarray
            Integer array of shape ``(N,)`` with tissue-type labels.
        """
        h, w = tissue_mask.shape[:2]
        labels = np.full(len(cell_coords), self.background_class, dtype=np.int64)
        for i, (x, y) in enumerate(cell_coords):
            xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
            labels[i] = int(tissue_mask[yi, xi])
        return labels

    def build_region_context(
        self,
        tissue_mask: np.ndarray,
        cell_coords: np.ndarray,
        radius: int = 32,
    ) -> np.ndarray:
        """Build a per-cell local tissue-type histogram as contextual features.

        Parameters
        ----------
        tissue_mask:
            Integer segmentation mask of shape ``(H, W)``.
        cell_coords:
            Array of shape ``(N, 2)`` with ``(x, y)`` cell centres.
        radius:
            Pixel radius of the local neighbourhood to consider.

        Returns
        -------
        np.ndarray
            Float array of shape ``(N, num_classes)`` — normalised histogram
            of tissue-class pixel frequencies within the local neighbourhood.
        """
        h, w = tissue_mask.shape[:2]
        histograms = np.zeros((len(cell_coords), self.num_classes), dtype=np.float32)
        for i, (x, y) in enumerate(cell_coords):
            x0 = max(0, int(x) - radius)
            y0 = max(0, int(y) - radius)
            x1 = min(w, int(x) + radius + 1)
            y1 = min(h, int(y) + radius + 1)
            region = tissue_mask[y0:y1, x0:x1].ravel()
            for c in range(self.num_classes):
                histograms[i, c] = np.sum(region == c)
            total = histograms[i].sum()
            if total > 0:
                histograms[i] /= total
        return histograms
