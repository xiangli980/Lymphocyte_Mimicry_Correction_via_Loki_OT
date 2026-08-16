"""
Cost matrix construction for Loki OT.

Builds pairwise cost matrices between source (detected) cells and target
(reference) cell distributions used in the unbalanced optimal transport
correction step.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def euclidean_cost(
    source_features: np.ndarray,
    target_features: np.ndarray,
    normalise: bool = True,
) -> np.ndarray:
    """Compute the squared Euclidean cost matrix between two feature sets.

    Parameters
    ----------
    source_features:
        Array of shape ``(M, D)`` — detected cell embeddings.
    target_features:
        Array of shape ``(N, D)`` — reference cell embeddings.
    normalise:
        If ``True``, divide by the maximum value so costs lie in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Cost matrix of shape ``(M, N)``.
    """
    C = cdist(source_features, target_features, metric="sqeuclidean")
    if normalise and C.max() > 0:
        C /= C.max()
    return C.astype(np.float64)


def mixed_cost(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_tissue: np.ndarray,
    target_tissue: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Combine appearance and tissue-context dissimilarity into one cost matrix.

    The mixed cost is a convex combination of the normalised feature distance
    and the tissue-context label mismatch:

    .. math::

        C_{ij} = \\alpha \\cdot C^{\\text{feat}}_{ij}
                 + (1-\\alpha) \\cdot \\mathbf{1}[t_i \\neq t_j]

    Parameters
    ----------
    source_features:
        Appearance embeddings for source cells ``(M, D)``.
    target_features:
        Appearance embeddings for target cells ``(N, D)``.
    source_tissue:
        Tissue-type label per source cell ``(M,)``.
    target_tissue:
        Tissue-type label per target cell ``(N,)``.
    alpha:
        Weight for the appearance term; ``1 - alpha`` weights tissue mismatch.

    Returns
    -------
    np.ndarray
        Cost matrix of shape ``(M, N)``.
    """
    C_feat = euclidean_cost(source_features, target_features, normalise=True)
    tissue_mismatch = (source_tissue[:, None] != target_tissue[None, :]).astype(np.float64)
    return alpha * C_feat + (1.0 - alpha) * tissue_mismatch
