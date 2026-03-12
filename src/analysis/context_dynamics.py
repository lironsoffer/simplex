"""
Context dynamics analysis: how belief state geometry changes with context position.

Main analysis:
    - R² for π and η_k as a function of context position t ∈ [0, L-1]
    - Verification of synchronisation horizon N*
    - Effective dimensionality vs context position
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import torch
from transformer_lens import HookedTransformer

from src.data.mess3 import Mess3HMM, compute_synchronisation_horizon
from src.analysis.regression import (
    extract_residual_stream_all_layers,
    fit_belief_regression,
)
from src.analysis.pca import compute_pca


def r2_vs_position(
    model: HookedTransformer,
    tokens: torch.Tensor,
    pi_targets: np.ndarray,
    eta_targets: np.ndarray,
    layer: int,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    """
    Compute R² for π and η_k regression at every context position for one layer.

    Args:
        model: trained HookedTransformer
        tokens: (N, L) int64
        pi_targets: (N, L, K) float32
        eta_targets: (N, L, K, 3) float32
        layer: which layer's residual stream to use
        batch_size: batch size for activation extraction
        device: torch device

    Returns:
        dict with:
            positions: (L,) array of position indices
            pi_r2: (L,) R² for π at each position
            eta_r2: (L, K) R² for each η_k at each position
    """
    if device is None:
        device = str(next(model.parameters()).device)

    N, L, K = pi_targets.shape

    # Extract activations for the given layer
    all_acts = extract_residual_stream_all_layers(model, tokens, batch_size, device)
    acts = all_acts[f"layer_{layer}_resid_post"]  # (N, L, d_model)

    pi_r2 = np.zeros(L)
    eta_r2 = np.zeros((L, K))

    for pos in range(L):
        acts_pos = acts[:, pos, :]           # (N, d_model)
        pi_pos = pi_targets[:, pos, :]       # (N, K)
        eta_pos = eta_targets[:, pos, :, :]  # (N, K, 3)

        pi_result = fit_belief_regression(acts_pos, pi_pos, n_subspace_components=K - 1)
        pi_r2[pos] = pi_result.r2

        for k in range(K):
            eta_k_result = fit_belief_regression(acts_pos, eta_pos[:, k, :], n_subspace_components=2)
            eta_r2[pos, k] = eta_k_result.r2

    return {
        "positions": np.arange(L),
        "pi_r2": pi_r2,
        "eta_r2": eta_r2,
    }


def r2_vs_position_all_layers(
    model: HookedTransformer,
    tokens: torch.Tensor,
    pi_targets: np.ndarray,
    eta_targets: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    """
    Compute R² vs position for all layers.

    Returns:
        dict with:
            pi_r2: (n_layers, L)
            eta_r2: (n_layers, L, K)
    """
    if device is None:
        device = str(next(model.parameters()).device)

    n_layers = model.cfg.n_layers
    N, L, K = pi_targets.shape

    all_acts = extract_residual_stream_all_layers(model, tokens, batch_size, device)

    pi_r2_all = np.zeros((n_layers, L))
    eta_r2_all = np.zeros((n_layers, L, K))

    for layer in range(n_layers):
        acts = all_acts[f"layer_{layer}_resid_post"]  # (N, L, d_model)

        for pos in range(L):
            acts_pos = acts[:, pos, :]           # (N, d_model)
            pi_pos = pi_targets[:, pos, :]       # (N, K)
            eta_pos = eta_targets[:, pos, :, :]  # (N, K, 3)

            pi_result = fit_belief_regression(acts_pos, pi_pos, n_subspace_components=K - 1)
            pi_r2_all[layer, pos] = pi_result.r2

            for k in range(K):
                eta_k_result = fit_belief_regression(
                    acts_pos, eta_pos[:, k, :], n_subspace_components=2
                )
                eta_r2_all[layer, pos, k] = eta_k_result.r2

    return {
        "pi_r2": pi_r2_all,
        "eta_r2": eta_r2_all,
        "n_layers": n_layers,
        "L": L,
        "K": K,
    }


def dimensionality_vs_position(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    cev_threshold: float = 0.95,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Compute effective dimensionality of residual stream at each context position.

    Args:
        model: trained HookedTransformer
        tokens: (N, L) int64
        layer: which layer's residual stream
        cev_threshold: CEV threshold for dimensionality
        batch_size: batch size
        device: torch device

    Returns:
        cev_dims: (L,) effective dimensionality at each position
    """
    if device is None:
        device = str(next(model.parameters()).device)

    N, L = tokens.shape

    all_acts = extract_residual_stream_all_layers(model, tokens, batch_size, device)
    acts = all_acts[f"layer_{layer}_resid_post"]  # (N, L, d_model)

    cev_dims = np.zeros(L, dtype=int)
    for pos in range(L):
        acts_pos = acts[:, pos, :]  # (N, d_model)
        pca_result = compute_pca(acts_pos, cev_threshold=cev_threshold)
        cev_dims[pos] = pca_result.cev_dim

    return cev_dims


def plot_r2_vs_position(
    positions: np.ndarray,
    pi_r2: np.ndarray,
    eta_r2: np.ndarray,
    n_star: float | None = None,
    component_names: list[str] | None = None,
    title: str = "R² vs Context Position",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot R² for π and η_k as a function of context position.

    Args:
        positions: (L,) position indices
        pi_r2: (L,) R² for π
        eta_r2: (L, K) R² for each η_k
        n_star: synchronisation horizon (plotted as vertical line)
        component_names: list of K component names
        title: plot title
        ax: existing axes (optional)
        save_path: save figure to path (optional)

    Returns:
        fig: matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    K = eta_r2.shape[1]
    if component_names is None:
        component_names = [f"Component {k}" for k in range(K)]

    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    ax.plot(positions + 1, pi_r2, "k-o", ms=5, linewidth=2, label="π (meta-belief)")

    for k in range(K):
        ax.plot(
            positions + 1, eta_r2[:, k],
            "-s", color=colors[k % len(colors)], ms=4, linewidth=1.5,
            label=f"η_{k} ({component_names[k]})"
        )

    if n_star is not None:
        ax.axvline(n_star, color="red", linestyle="--", alpha=0.7, label=f"N* = {n_star:.1f}")

    ax.set_xlabel("Context position (tokens seen)")
    ax.set_ylabel("R² (linear decodability)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, len(positions) + 0.5)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compute_nstar_analytical(
    components: list[Mess3HMM],
    n_sequences: int = 500,
    seq_length: int = 200,
) -> float:
    """
    Compute synchronisation horizon N* empirically.

    N* = 1 / min_{k != k'} D_KL(P_k || P_k')

    where D_KL is the per-token KL divergence rate between HMM processes.
    (Note: 1-gram marginals are identical for all Mess3 components, so we use
    empirical log-likelihood ratios on longer sequences.)

    Args:
        components: list of Mess3HMM instances
        n_sequences: sequences for empirical KL estimation
        seq_length: sequence length for KL estimation

    Returns:
        N*: synchronisation horizon (should be < context window length)
    """
    return compute_synchronisation_horizon(components, n_sequences=n_sequences, seq_length=seq_length)
