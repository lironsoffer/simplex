"""
Training dynamics analysis: geometry evolution across checkpoints.

Loads saved checkpoints and runs geometry analysis at each training step
to track when belief subspaces emerge.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from src.model.transformer import build_model
from src.training.train import load_checkpoint
from src.analysis.regression import (
    extract_residual_stream_all_layers,
    fit_belief_regression,
)


def get_checkpoint_paths(checkpoint_dir: str) -> list[tuple[int, str]]:
    """
    Find all checkpoint files in a directory, sorted by step.

    Args:
        checkpoint_dir: path to checkpoint directory

    Returns:
        list of (step, path) tuples, sorted by step
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = []

    for f in checkpoint_path.glob("checkpoint_step_*.pt"):
        step_str = f.stem.split("_step_")[-1]
        try:
            step = int(step_str)
            checkpoints.append((step, str(f)))
        except ValueError:
            continue

    return sorted(checkpoints, key=lambda x: x[0])


def analyze_checkpoint(
    model: HookedTransformer,
    tokens: torch.Tensor,
    pi_targets: np.ndarray,
    eta_targets: np.ndarray,
    layer: int = -1,
    position: int = -1,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    """
    Run geometry analysis at a specific layer and position.

    Args:
        model: HookedTransformer with loaded checkpoint weights
        tokens: (N, L) int64
        pi_targets: (N, L, K) float32
        eta_targets: (N, L, K, 3) float32
        layer: which layer to analyse (-1 for last layer)
        position: which context position (-1 for last position)
        batch_size: batch size for activation extraction
        device: torch device

    Returns:
        dict with R² for π and each η_k
    """
    if device is None:
        device = str(next(model.parameters()).device)

    n_layers = model.cfg.n_layers
    N, L, K = pi_targets.shape

    if layer < 0:
        layer = n_layers + layer
    if position < 0:
        position = L + position

    all_acts = extract_residual_stream_all_layers(model, tokens, batch_size, device)
    acts = all_acts[f"layer_{layer}_resid_post"][:, position, :]  # (N, d_model)

    pi_pos = pi_targets[:, position, :]         # (N, K)
    eta_pos = eta_targets[:, position, :, :]    # (N, K, 3)

    pi_result = fit_belief_regression(acts, pi_pos, n_subspace_components=K - 1)
    eta_r2 = []
    for k in range(K):
        eta_k_result = fit_belief_regression(acts, eta_pos[:, k, :], n_subspace_components=2)
        eta_r2.append(eta_k_result.r2)

    return {
        "pi_r2": pi_result.r2,
        "eta_r2": eta_r2,
        "layer": layer,
        "position": position,
    }


def run_training_dynamics(
    checkpoint_dir: str,
    tokens: torch.Tensor,
    pi_targets: np.ndarray,
    eta_targets: np.ndarray,
    model_kwargs: dict | None = None,
    layer: int = -1,
    position: int = -1,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    """
    Run geometry analysis across all checkpoints to track training dynamics.

    Args:
        checkpoint_dir: directory containing checkpoint files
        tokens: (N, L) int64
        pi_targets: (N, L, K) float32
        eta_targets: (N, L, K, 3) float32
        model_kwargs: kwargs for build_model (uses defaults if None)
        layer: which layer to analyse (-1 for last)
        position: which context position (-1 for last)
        batch_size: batch size
        device: torch device

    Returns:
        dynamics dict with steps and R² over training
    """
    if model_kwargs is None:
        model_kwargs = {}

    checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    model = build_model(**model_kwargs)
    if device:
        model = model.to(device)

    steps = []
    pi_r2_over_time = []
    eta_r2_over_time = []

    for step, ckpt_path in checkpoint_paths:
        model, _, _ = load_checkpoint(model, ckpt_path, device)
        model.eval()

        result = analyze_checkpoint(
            model, tokens, pi_targets, eta_targets,
            layer=layer, position=position,
            batch_size=batch_size, device=device,
        )

        steps.append(step)
        pi_r2_over_time.append(result["pi_r2"])
        eta_r2_over_time.append(result["eta_r2"])
        print(f"Step {step:6d}: π R²={result['pi_r2']:.3f}, η R²={result['eta_r2']}")

    return {
        "steps": np.array(steps),
        "pi_r2": np.array(pi_r2_over_time),
        "eta_r2": np.array(eta_r2_over_time),
        "layer": layer,
        "position": position,
    }


def plot_training_dynamics(
    dynamics: dict,
    component_names: list[str] | None = None,
    title: str = "Belief Subspace Emergence over Training",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot R² for π and η_k over training steps.

    Args:
        dynamics: output of run_training_dynamics
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

    steps = dynamics["steps"]
    pi_r2 = dynamics["pi_r2"]
    eta_r2 = dynamics["eta_r2"]
    K = eta_r2.shape[1] if eta_r2.ndim > 1 else 1

    if component_names is None:
        component_names = [f"Component {k}" for k in range(K)]

    colors = ["steelblue", "darkorange", "forestgreen"]

    ax.plot(steps, pi_r2, "k-o", ms=5, linewidth=2, label="π (meta-belief)")
    for k in range(K):
        ax.plot(
            steps, eta_r2[:, k],
            "-s", color=colors[k % len(colors)], ms=4, linewidth=1.5,
            label=f"η_{k} ({component_names[k]})"
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("R² (linear decodability)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def find_phase_transition(
    steps: np.ndarray,
    r2_values: np.ndarray,
    threshold: float = 0.9,
) -> int | None:
    """
    Find the training step at which R² first exceeds a threshold.

    Args:
        steps: (T,) training steps
        r2_values: (T,) R² values over training
        threshold: R² threshold for detecting phase transition

    Returns:
        step: first step where R² > threshold, or None if never reached
    """
    indices = np.where(r2_values >= threshold)[0]
    if len(indices) == 0:
        return None
    return int(steps[indices[0]])
