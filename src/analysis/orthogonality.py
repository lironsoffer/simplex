"""
Subspace orthogonality analysis.

Methods:
    - subspace_overlap: measure angle between two subspaces (Grassmann distance)
    - projection_test: regress out one subspace, check if other survives
    - vary_one_analysis: fix component, vary context position, track subspace stability
    - pairwise_overlap_matrix: full overlap matrix between all belief subspaces
"""

import numpy as np
from dataclasses import dataclass

from src.analysis.regression import (
    fit_belief_regression,
    project_out_subspace,
)


@dataclass
class ProjectionTestResult:
    """Result of projecting out one subspace and re-regressing another."""
    r2_before: float   # R² before projection
    r2_after: float    # R² after projecting out the other subspace
    r2_drop: float     # r2_before - r2_after (positive = subspaces share info)


def subspace_overlap(
    basis_A: np.ndarray,
    basis_B: np.ndarray,
) -> float:
    """
    Compute the subspace overlap between two subspaces.

    Uses the Frobenius norm of the product of projection matrices:
        overlap = ||P_A P_B||_F / sqrt(rank_A * rank_B)

    Where P_A = A^T (A A^T)^{-1} A is the orthogonal projector onto subspace A.
    For orthonormal basis rows: P_A = A^T A.

    Args:
        basis_A: (r_A, d_model) row vectors spanning subspace A
        basis_B: (r_B, d_model) row vectors spanning subspace B

    Returns:
        overlap: scalar in [0, 1], 0 = orthogonal, 1 = identical subspaces
    """
    # Orthonormalise both bases via QR
    Q_A = _orthonormalise_rows(basis_A)  # (r_A, d_model)
    Q_B = _orthonormalise_rows(basis_B)  # (r_B, d_model)

    # Singular values of Q_A @ Q_B^T = cos of principal angles
    M = Q_A @ Q_B.T  # (r_A, r_B)
    singular_values = np.linalg.svd(M, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)

    # Frobenius norm of product of projectors = sqrt(sum of squared singular values)
    # Normalised by sqrt(r_A * r_B)
    r_A = Q_A.shape[0]
    r_B = Q_B.shape[0]
    overlap = np.sqrt(np.sum(singular_values ** 2)) / np.sqrt(r_A * r_B)

    return float(overlap)


def _orthonormalise_rows(basis: np.ndarray) -> np.ndarray:
    """Return orthonormal row vectors spanning the same subspace."""
    Q, _ = np.linalg.qr(basis.T)  # Q: (d_model, rank)
    rank = min(basis.shape)
    return Q[:, :rank].T  # (rank, d_model)


def pairwise_overlap_matrix(
    subspace_bases: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise subspace overlap for all pairs of subspaces.

    Args:
        subspace_bases: dict mapping name → (n_comp, d_model) basis

    Returns:
        overlap_matrix: (n_subspaces, n_subspaces) symmetric matrix
        names: list of subspace names in matrix order
    """
    names = list(subspace_bases.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                overlap = subspace_overlap(
                    subspace_bases[name_i],
                    subspace_bases[name_j],
                )
                matrix[i, j] = overlap
                matrix[j, i] = overlap

    return matrix, names


def projection_test(
    activations: np.ndarray,
    basis_remove: np.ndarray,
    targets_keep: np.ndarray,
) -> ProjectionTestResult:
    """
    Test whether regressing out one subspace degrades another target's R².

    Procedure:
        1. Fit regression from activations → targets_keep; record R² before
        2. Project out basis_remove from activations
        3. Re-fit regression on projected activations → targets_keep; record R² after

    A large R² drop (H1) means the subspaces share information.
    A small R² drop (H2) means the subspaces are orthogonal.

    Args:
        activations: (N, d_model)
        basis_remove: (n_comp, d_model) subspace to remove
        targets_keep: (N, n_targets) targets we want to preserve

    Returns:
        ProjectionTestResult with r2_before, r2_after, r2_drop
    """
    # Before projection
    result_before = fit_belief_regression(activations, targets_keep)
    r2_before = result_before.r2

    # Project out the subspace
    acts_projected = project_out_subspace(activations, basis_remove)

    # After projection
    result_after = fit_belief_regression(acts_projected, targets_keep)
    r2_after = result_after.r2

    return ProjectionTestResult(
        r2_before=r2_before,
        r2_after=r2_after,
        r2_drop=r2_before - r2_after,
    )


def full_orthogonality_analysis(
    activations: np.ndarray,
    pi_targets: np.ndarray,
    eta_targets: np.ndarray,
    K: int,
) -> dict:
    """
    Run the complete orthogonality analysis suite.

    For each pair (π subspace vs η_k subspace), compute:
        1. Subspace overlap
        2. Projection test: remove π, check η_k R²
        3. Projection test: remove η_k, check π R²
        4. Cross-tests: remove η_j, check η_k R²

    Args:
        activations: (N, d_model) residual stream at one layer+position
        pi_targets: (N, K) meta-belief targets
        eta_targets: (N, K, 3) within-component belief targets
        K: number of components

    Returns:
        results dict with overlap matrix and projection tests
    """
    # Fit all regressions to extract subspace bases
    pi_result = fit_belief_regression(activations, pi_targets, n_subspace_components=K - 1)
    eta_results = [
        fit_belief_regression(activations, eta_targets[:, k, :], n_subspace_components=2)
        for k in range(K)
    ]

    # Collect bases
    subspace_bases = {"pi": pi_result.subspace_basis}
    for k in range(K):
        subspace_bases[f"eta_{k}"] = eta_results[k].subspace_basis

    # Pairwise overlap matrix
    overlap_matrix, names = pairwise_overlap_matrix(subspace_bases)

    # Projection tests: remove π subspace, check each η_k
    projection_pi_into_eta = []
    for k in range(K):
        pt = projection_test(
            activations,
            basis_remove=pi_result.subspace_basis,
            targets_keep=eta_targets[:, k, :],
        )
        projection_pi_into_eta.append(pt)

    # Projection tests: remove each η_k subspace, check π
    projection_eta_into_pi = []
    for k in range(K):
        pt = projection_test(
            activations,
            basis_remove=eta_results[k].subspace_basis,
            targets_keep=pi_targets,
        )
        projection_eta_into_pi.append(pt)

    # Cross-projection matrix: remove η_j subspace, check η_k R² drop for all j≠k pairs
    # cross_projection_matrix[j, k] = R² drop in η_k after removing η_j; diagonal = 0
    # cross_projection_details[j][k] = ProjectionTestResult for j≠k, None if j==k
    cross_projection_matrix = np.zeros((K, K))
    cross_projection_details = [[None] * K for _ in range(K)]

    for j in range(K):
        for k in range(K):
            if j != k:
                pt = projection_test(
                    activations,
                    basis_remove=eta_results[j].subspace_basis,
                    targets_keep=eta_targets[:, k, :],
                )
                cross_projection_matrix[j, k] = pt.r2_drop
                cross_projection_details[j][k] = pt

    return {
        "subspace_bases": subspace_bases,
        "overlap_matrix": overlap_matrix,
        "overlap_names": names,
        "pi_r2": pi_result.r2,
        "eta_r2": [r.r2 for r in eta_results],
        "projection_pi_into_eta": projection_pi_into_eta,
        "projection_eta_into_pi": projection_eta_into_pi,
        "cross_projection_matrix": cross_projection_matrix,
        "cross_projection_details": cross_projection_details,
    }


def vary_one_analysis(
    activations_by_component: dict[int, np.ndarray],
    eta_targets_by_component: dict[int, np.ndarray],
    K: int,
) -> dict:
    """
    Vary-one analysis from Shai et al. 2025.

    For each component k, fit the η_k subspace using sequences from component k only.
    Then check if the subspace is stable (consistent across different position subsets)
    and how much it overlaps with other components' subspaces.

    Args:
        activations_by_component: dict {k: (N_k, d_model)} activations for each component
        eta_targets_by_component: dict {k: (N_k, 3)} η_k targets for each component
        K: number of components

    Returns:
        vary_one_results dict with per-component subspace overlap
    """
    # Fit per-component subspaces
    per_component_bases = {}
    per_component_r2 = {}

    for k in range(K):
        acts = activations_by_component[k]
        eta_k = eta_targets_by_component[k]
        result = fit_belief_regression(acts, eta_k, n_subspace_components=2)
        per_component_bases[k] = result.subspace_basis
        per_component_r2[k] = result.r2

    # Pairwise overlap between component subspaces
    cross_overlap = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                cross_overlap[i, j] = 1.0
            elif i < j:
                overlap = subspace_overlap(
                    per_component_bases[i],
                    per_component_bases[j],
                )
                cross_overlap[i, j] = overlap
                cross_overlap[j, i] = overlap

    return {
        "per_component_bases": per_component_bases,
        "per_component_r2": per_component_r2,
        "cross_component_overlap": cross_overlap,
    }
