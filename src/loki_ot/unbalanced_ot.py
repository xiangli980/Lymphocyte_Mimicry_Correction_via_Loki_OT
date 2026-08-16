"""
Loki OT — Unbalanced Optimal Transport solver.

Implements the Kullback–Leibler-regularised unbalanced OT plan (Sinkhorn-like
scaling) used to correct lymphocyte mimicry detections.

Reference:
    Chizat et al., "Scaling Algorithms for Unbalanced Optimal Transport Problems",
    Mathematics of Computation, 2018.
"""

from __future__ import annotations

import numpy as np


def sinkhorn_kl_unbalanced(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    epsilon: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve the KL-regularised unbalanced OT problem via Sinkhorn iterations.

    Minimises:

    .. math::

        \\min_{T \\geq 0}
            \\langle C, T \\rangle
            + \\epsilon \\, \\text{KL}(T | a \\otimes b)
            + \\rho \\, \\text{KL}(T\\mathbf{1} | a)
            + \\rho \\, \\text{KL}(T^\\top\\mathbf{1} | b)

    Parameters
    ----------
    a:
        Source marginal histogram of shape ``(M,)``. Does **not** need to
        sum to 1 (unbalanced).
    b:
        Target marginal histogram of shape ``(N,)``.
    C:
        Cost matrix of shape ``(M, N)``.
    epsilon:
        Entropic regularisation strength.
    rho:
        KL marginal relaxation strength.  Large ``rho`` → balanced OT.
    max_iter:
        Maximum number of Sinkhorn iterations.
    tol:
        Convergence tolerance on the transport plan change.

    Returns
    -------
    np.ndarray
        Optimal transport plan ``T`` of shape ``(M, N)``.
    """
    M, N = C.shape
    assert len(a) == M and len(b) == N, "Marginal sizes must match cost matrix dimensions."

    # Log-domain Sinkhorn for numerical stability.
    lam = rho / (rho + epsilon)  # scaling exponent for unbalanced updates

    log_K = -C / epsilon  # (M, N)
    log_u = np.zeros(M)  # dual variable u (log scale)
    log_v = np.zeros(N)  # dual variable v (log scale)

    T_prev = np.zeros((M, N))

    for _ in range(max_iter):
        # Update v: soft marginal on b.
        log_KTu = _log_sum_exp(log_K + log_u[:, None], axis=0)  # (N,)
        log_v = lam * (np.log(b + 1e-300) - log_KTu)

        # Update u: soft marginal on a.
        log_Kv = _log_sum_exp(log_K + log_v[None, :], axis=1)  # (M,)
        log_u = lam * (np.log(a + 1e-300) - log_Kv)

        # Transport plan in primal.
        log_T = log_K + log_u[:, None] + log_v[None, :]
        T = np.exp(log_T)

        if np.max(np.abs(T - T_prev)) < tol:
            break
        T_prev = T

    return T


def _log_sum_exp(X: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis``."""
    X_max = X.max(axis=axis, keepdims=True)
    return np.log(np.exp(X - X_max).sum(axis=axis)) + X_max.squeeze(axis=axis)


def compute_transport_assignment(
    T: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Convert a soft transport plan to hard cell-level assignments.

    Each source cell ``i`` is assigned to the target class ``j*`` that
    receives the maximum transport mass, subject to the mass exceeding
    ``threshold``.

    Parameters
    ----------
    T:
        Transport plan of shape ``(M, N)``.
    threshold:
        Minimum mass required; cells below threshold are marked ``-1``
        (unassigned / filtered out as mimickers).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(M,)`` with target indices.  ``-1`` means
        the cell is discarded as a mimicker.
    """
    assignment = T.argmax(axis=1)
    max_mass = T.max(axis=1)
    assignment[max_mass <= threshold] = -1
    return assignment
