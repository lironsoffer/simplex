"""
Linear regression from residual stream activations to belief state targets.

Methods:
    - fit_belief_regression: fit linear map activations → belief targets
    - fit_belief_regression_oos: fit with held-out test split for OOS R²
    - extract_subspace_basis: top singular vectors of regression weight matrix
    - extract_residual_stream_all_layers: extract activations at all layers
    - project_out_subspace: project out a subspace from activations
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass
class RegressionResult:
    """Results from fitting a linear regression."""
    r2: float                     # R² on the fitting data
    weights: np.ndarray           # Weight matrix (d_model, n_targets)
    bias: np.ndarray              # Bias vector (n_targets,)
    subspace_basis: np.ndarray    # Top singular vectors of W, shape (n_components, d_model)
    singular_values: np.ndarray   # Singular values of W


def fit_belief_regression(
    activations: np.ndarray,
    targets: np.ndarray,
    alpha: float = 1e-4,
    n_subspace_components: int = 4,
) -> RegressionResult:
    """
    Fit a linear regression from residual stream activations to belief targets.

    Uses Ridge regression for numerical stability. Extracts the top singular
    vectors of the weight matrix as the subspace basis.

    Args:
        activations: (N, d_model) residual stream activations
        targets: (N, n_targets) belief state targets
        alpha: Ridge regularisation strength
        n_subspace_components: number of singular vectors to extract

    Returns:
        RegressionResult with R², weights, and subspace basis
    """
    N, d_model = activations.shape
    n_targets = targets.shape[1]

    # Normalise activations
    act_mean = activations.mean(axis=0, keepdims=True)
    act_std = activations.std(axis=0, keepdims=True) + 1e-8
    activations_norm = (activations - act_mean) / act_std

    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(activations_norm, targets)

    predictions = reg.predict(activations_norm)
    r2 = _r2_score(targets, predictions)

    W = reg.coef_  # (n_targets, d_model)
    b = reg.intercept_  # (n_targets,)

    # Extract subspace: top singular vectors of W (rows are target-space directions)
    n_comp = min(n_subspace_components, min(W.shape))
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp)
        svd.fit(W)
        basis = svd.components_  # (n_comp, d_model) — principal directions in activation space
        singular_values = svd.singular_values_
    else:
        basis = np.zeros((1, d_model))
        singular_values = np.zeros(1)

    return RegressionResult(
        r2=r2,
        weights=W.T,         # (d_model, n_targets)
        bias=b,
        subspace_basis=basis,       # (n_comp, d_model)
        singular_values=singular_values,
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def extract_residual_stream_all_layers(
    model: HookedTransformer,
    tokens: torch.Tensor,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract residual stream activations at all layers for all sequences.

    Args:
        model: HookedTransformer
        tokens: (N, L) int64 token tensor
        batch_size: batch size for forward passes
        device: torch device

    Returns:
        cache: dict mapping "layer_{i}_resid_post" → (N, L, d_model) float32 numpy array
    """
    if device is None:
        device = str(next(model.parameters()).device)

    n_layers = model.cfg.n_layers
    N, L = tokens.shape
    d_model = model.cfg.d_model

    # Pre-allocate
    all_activations = {
        f"layer_{i}_resid_post": np.zeros((N, L, d_model), dtype=np.float32)
        for i in range(n_layers)
    }

    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_tokens = tokens[start:end].to(device)

            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda name: "hook_resid_post" in name,
                return_type="logits",
            )

            for i in range(n_layers):
                hook_name = f"blocks.{i}.hook_resid_post"
                all_activations[f"layer_{i}_resid_post"][start:end] = (
                    cache[hook_name].float().cpu().numpy()
                )

    return all_activations



def fit_belief_regression_oos(
    activations: np.ndarray,
    targets: np.ndarray,
    alpha: float = 1e-4,
    n_subspace_components: int = 4,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[RegressionResult, float]:
    """
    Fit a linear regression from residual stream activations to belief targets,
    evaluating R² on a held-out test split to avoid in-sample inflation.

    Args:
        activations: (N, d_model) residual stream activations
        targets: (N, n_targets) belief state targets
        alpha: Ridge regularisation strength
        n_subspace_components: number of singular vectors to extract
        test_fraction: fraction of data to hold out for test evaluation
        seed: random seed for reproducible train/test split

    Returns:
        (result, r2_test): RegressionResult fitted on train split (with in-sample
        train R²), and the out-of-sample R² on the test split.
    """
    N = activations.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    n_test = max(1, int(np.floor(N * test_fraction)))
    n_train = N - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_activations = activations[train_idx]
    train_targets = targets[train_idx]
    test_activations = activations[test_idx]
    test_targets = targets[test_idx]

    # Fit on train split
    result = fit_belief_regression(
        train_activations,
        train_targets,
        alpha=alpha,
        n_subspace_components=n_subspace_components,
    )

    # Compute normalisation stats from train (same as fit_belief_regression does internally)
    act_mean = train_activations.mean(axis=0, keepdims=True)
    act_std = train_activations.std(axis=0, keepdims=True) + 1e-8

    # Reconstruct the Ridge regressor from stored weights/bias to predict on test
    # result.weights is (d_model, n_targets), so W = weights.T is (n_targets, d_model)
    W = result.weights.T   # (n_targets, d_model)
    b = result.bias        # (n_targets,)

    test_activations_norm = (test_activations - act_mean) / act_std
    test_predictions = test_activations_norm @ W.T + b  # (n_test, n_targets)

    r2_test = _r2_score(test_targets, test_predictions)

    return result, r2_test



def project_out_subspace(
    activations: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Project out a subspace from activations.

    Removes the components along the given basis directions.

    Args:
        activations: (N, d_model)
        basis: (n_comp, d_model), orthonormal rows

    Returns:
        projected: (N, d_model) with subspace removed
    """
    # Orthonormalise basis via QR
    Q, _ = np.linalg.qr(basis.T)  # Q: (d_model, n_comp)
    Q = Q[:, :basis.shape[0]]

    # Project out
    coeff = activations @ Q        # (N, n_comp)
    return activations - coeff @ Q.T  # (N, d_model)
