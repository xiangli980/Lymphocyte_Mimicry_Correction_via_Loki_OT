"""
End-to-end lymphocyte mimicry correction pipeline.

Ties together tissue reasoning and Loki OT to produce corrected cell labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..tissue_reasoning.region_extractor import RegionExtractor
from ..tissue_reasoning.tissue_classifier import TissueClassifier
from ..loki_ot.cost_matrix import mixed_cost
from ..loki_ot.unbalanced_ot import sinkhorn_kl_unbalanced, compute_transport_assignment


_DEFAULT_CONFIG: dict[str, Any] = {
    "tissue_reasoning": {
        "backbone": "resnet50",
        "pretrained": True,
        "patch_size": 64,
        "num_tissue_classes": 5,
        "context_radius": 32,
    },
    "loki_ot": {
        "epsilon": 0.1,
        "rho": 1.0,
        "max_iter": 1000,
        "tol": 1e-9,
        "alpha": 0.5,
        "mass_threshold": 0.0,
    },
}


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return _DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Merge with defaults so missing keys fall back gracefully.
    merged = {**_DEFAULT_CONFIG}
    for section, values in cfg.items():
        if section in merged and isinstance(merged[section], dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return merged


class CorrectionPipeline:
    """Lymphocyte mimicry correction pipeline.

    Parameters
    ----------
    config:
        Path to a YAML configuration file, or ``None`` to use built-in
        defaults.
    device:
        Torch device string.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        config: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = _load_config(config)
        tr_cfg = self.cfg["tissue_reasoning"]
        self.extractor = RegionExtractor(
            backbone=tr_cfg["backbone"],
            pretrained=tr_cfg["pretrained"],
            device=device,
            patch_size=tr_cfg["patch_size"],
        )
        self.classifier = TissueClassifier(
            num_classes=tr_cfg["num_tissue_classes"],
        )

    def run(
        self,
        wsi_image: np.ndarray,
        tissue_mask: np.ndarray,
        cell_coords: np.ndarray,
        reference_features: np.ndarray,
        reference_tissue_labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run the full correction pipeline.

        Parameters
        ----------
        wsi_image:
            Whole-slide image as ``(H, W, 3)`` uint8 NumPy array.
        tissue_mask:
            Tissue segmentation mask ``(H, W)`` with integer class labels.
        cell_coords:
            Detected cell centres ``(N, 2)`` in ``(x, y)`` pixel coordinates.
        reference_features:
            Pre-computed reference lymphocyte embeddings ``(K, D)``.
        reference_tissue_labels:
            Tissue-type labels for each reference cell ``(K,)``.

        Returns
        -------
        dict with keys:

        - ``"assignments"``: hard transport assignment per detected cell ``(N,)``
        - ``"transport_plan"``: soft transport plan ``(N, K)``
        - ``"tissue_labels"``: tissue-type label per detected cell ``(N,)``
        - ``"cell_features"``: feature embedding per detected cell ``(N, D)``
        """
        tr_cfg = self.cfg["tissue_reasoning"]
        ot_cfg = self.cfg["loki_ot"]

        # 1. Extract appearance features for each detected cell.
        cell_features = self.extractor.extract_cell_patches(wsi_image, cell_coords)

        # 2. Assign tissue-type context to each detected cell.
        tissue_labels = self.classifier.assign_tissue_labels(tissue_mask, cell_coords)

        # 3. Build the mixed cost matrix.
        C = mixed_cost(
            cell_features,
            reference_features,
            tissue_labels,
            reference_tissue_labels,
            alpha=ot_cfg["alpha"],
        )

        # 4. Uniform source / target marginals.
        N = len(cell_coords)
        K = len(reference_features)
        a = np.ones(N, dtype=np.float64) / N
        b = np.ones(K, dtype=np.float64) / K

        # 5. Solve unbalanced OT.
        T = sinkhorn_kl_unbalanced(
            a, b, C,
            epsilon=ot_cfg["epsilon"],
            rho=ot_cfg["rho"],
            max_iter=ot_cfg["max_iter"],
            tol=ot_cfg["tol"],
        )

        # 6. Hard assignment (mimickers get label -1).
        assignments = compute_transport_assignment(T, threshold=ot_cfg["mass_threshold"])

        return {
            "assignments": assignments,
            "transport_plan": T,
            "tissue_labels": tissue_labels,
            "cell_features": cell_features,
        }

    @classmethod
    def from_csv(
        cls,
        detection_csv: str | Path,
        config: str | Path | None = None,
        x_col: str = "x",
        y_col: str = "y",
        **kwargs: Any,
    ) -> tuple["CorrectionPipeline", np.ndarray]:
        """Convenience constructor that reads cell detections from a CSV file.

        Parameters
        ----------
        detection_csv:
            Path to a CSV with at least ``x_col`` and ``y_col`` columns.
        config:
            Optional YAML config path.
        x_col / y_col:
            Column names for cell-centre coordinates.

        Returns
        -------
        (pipeline, cell_coords)
        """
        df = pd.read_csv(detection_csv)
        coords = df[[x_col, y_col]].values.astype(np.float32)
        return cls(config=config, **kwargs), coords
