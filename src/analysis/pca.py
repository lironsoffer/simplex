"""
PCA-based dimensionality analysis and fractal visualisation.

Methods:
    - compute_pca: run PCA on activations, return explained variance
    - compute_cev: cumulative explained variance dimensionality measure
    - plot_fractal: scatter plot of activations in belief subspace vs MSP attractor
    - compute_msp_attractor: simulate the IFS to get ground-truth MSP fractal
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataclasses import dataclass

from src.data.mess3 import Mess3HMM
from src.analysis.regression import fit_belief_regression


@dataclass
class PCAResult:
    """Results from PCA analysis."""
    explained_variance_ratio: np.ndarray   # (n_components,)
    cumulative_variance: np.ndarray        # (n_components,)
    components: np.ndarray                  # (n_components, d_model)
    cev_dim: int                            # CEV dimensionality at threshold


def compute_pca(
    activations: np.ndarray,
    n_components: int | None = None,
    cev_threshold: float = 0.95,
) -> PCAResult:
    """
    Run PCA on activations and compute CEV dimensionality.

    Args:
        activations: (N, d_model)
        n_components: number of PCA components (default: min(N, d_model))
        cev_threshold: threshold for CEV dimensionality (default: 0.95)

    Returns:
        PCAResult with explained variance, components, CEV dim
    """
    N, d = activations.shape
    if n_components is None:
        n_components = min(N, d)

    pca = PCA(n_components=n_components)
    pca.fit(activations)

    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    # CEV dimensionality: smallest d such that sum of top-d variance > threshold
    indices = np.where(cumvar >= cev_threshold)[0]
    cev_dim = int(indices[0] + 1) if len(indices) > 0 else n_components

    return PCAResult(
        explained_variance_ratio=evr,
        cumulative_variance=cumvar,
        components=pca.components_,
        cev_dim=cev_dim,
    )


def compute_msp_attractor(
    component: Mess3HMM,
    n_iterations: int = 50000,
    n_warmup: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate the IFS to approximate the MSP (Mixed State Presentation) fractal attractor.

    The MSP is the unique attractor of the three contraction maps:
        f_a(eta) = eta @ T^(a) / (eta @ T^(a) @ 1)

    Args:
        component: Mess3HMM instance
        n_iterations: number of IFS iterations
        n_warmup: warmup iterations to discard
        seed: random seed

    Returns:
        attractor_points: (n_iterations, 3) belief states on the fractal
    """
    rng = np.random.default_rng(seed)
    eta = component.stationary_distribution.copy()

    total = n_iterations + n_warmup
    points = np.zeros((total, 3))

    for i in range(total):
        token = rng.integers(0, 3)
        T_a = component.transition_matrices[token]
        unnorm = eta @ T_a
        eta = unnorm / unnorm.sum()
        points[i] = eta

    return points[n_warmup:]  # Discard warmup


def simplex_to_2d(belief: np.ndarray) -> np.ndarray:
    """
    Project 3-simplex coordinates to 2D for visualisation.

    Uses barycentric coordinates on an equilateral triangle.

    Args:
        belief: (N, 3) or (3,) belief states summing to 1

    Returns:
        xy: (N, 2) or (2,) 2D coordinates
    """
    single = belief.ndim == 1
    if single:
        belief = belief[None]

    # Vertices of equilateral triangle
    v = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
    ])

    xy = belief @ v  # (N, 2)
    return xy[0] if single else xy


def plot_fractal(
    activations_k: np.ndarray,
    eta_k_targets: np.ndarray,
    component: Mess3HMM,
    component_name: str = "",
    color: str = "steelblue",
    n_attractor_points: int = 80000,
    axes=None,
    save_path=None,
) -> plt.Figure:
    """
    Side-by-side comparison of the MSP fractal (matching Figure 2 of Shai et al. 2025):
      Left panel  — ground-truth η_k belief states in 2D simplex (barycentric) coords.
      Right panel — regression-predicted η_k from residual stream, also in simplex coords.

    Both panels use the same barycentric coordinate system, so a visual shape match
    confirms the model has learned the belief geometry. This matches the paper's approach:
    apply the linear regression (acts → η_k), then plot predicted η_k in the simplex.

    Args:
        activations_k: (N, d_model) — activations for component k, all post-N* positions
            stacked (N = n_sequences_k × n_positions_used).
        eta_k_targets: (N, 3) — corresponding ground-truth η_k belief states.
        component: Mess3HMM for component k (used to simulate IFS attractor background).
        component_name: label for plot title.
        n_attractor_points: IFS iterations for background attractor reference.
        axes: pair of matplotlib Axes [ax_left, ax_right]; creates a new figure if None.
        save_path: optional file path to save.

    Returns:
        fig: matplotlib Figure.
    """
    if axes is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))
    else:
        ax_l, ax_r = axes
        fig = ax_l.figure

    alpha_val = getattr(component, "alpha", "?")
    x_val = getattr(component, "x", "?")
    n_pts = len(eta_k_targets)

    # Fit regression: activations → η_k
    reg_result = fit_belief_regression(activations_k, eta_k_targets, n_subspace_components=2)

    # Compute predicted η_k from activations
    act_mean = activations_k.mean(axis=0, keepdims=True)
    act_std  = activations_k.std(axis=0, keepdims=True) + 1e-8
    acts_norm = (activations_k - act_mean) / act_std
    predicted_eta = acts_norm @ reg_result.weights + reg_result.bias  # (N, 3)

    # Shared: IFS attractor background and simplex outline
    attractor_beliefs = compute_msp_attractor(component, n_attractor_points)
    att_2d = simplex_to_2d(attractor_beliefs)
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])

    # ── Left panel: ground-truth η_k ─────────────────────────────────────────
    ax_l.scatter(att_2d[:, 0], att_2d[:, 1],
                 s=0.2, c="lightgray", alpha=0.3, rasterized=True)
    gt_2d = simplex_to_2d(eta_k_targets)
    ax_l.scatter(gt_2d[:, 0], gt_2d[:, 1],
                 s=0.8, c="dimgray", alpha=0.5, rasterized=True)
    ax_l.plot(v[:, 0], v[:, 1], "k-", lw=0.8, alpha=0.4)
    ax_l.set_aspect("equal")
    ax_l.set_xticks([]); ax_l.set_yticks([])
    ax_l.set_title(f"Ground truth η_{component_name}\n({n_pts} belief states, simplex coords)",
                   fontsize=9)

    # ── Right panel: regression-predicted η_k (same simplex coords) ──────────
    ax_r.scatter(att_2d[:, 0], att_2d[:, 1],
                 s=0.2, c="lightgray", alpha=0.3, rasterized=True)
    pred_2d = simplex_to_2d(np.clip(predicted_eta, 0, None))
    ax_r.scatter(pred_2d[:, 0], pred_2d[:, 1],
                 s=0.8, c=color, alpha=0.5, rasterized=True)
    ax_r.plot(v[:, 0], v[:, 1], "k-", lw=0.8, alpha=0.4)
    ax_r.set_aspect("equal")
    ax_r.set_xticks([]); ax_r.set_yticks([])
    ax_r.set_title(
        f"Activations — Component {component_name} (α={alpha_val}, x={x_val})\n"
        f"R²={reg_result.r2:.3f}, regression output in simplex coords",
        fontsize=9,
    )
    ax_r.set_ylabel("Subspace direction 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pca_explained_variance(
    pca_result: PCAResult,
    title: str = "PCA Cumulative Explained Variance",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot cumulative explained variance from PCA."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    n = len(pca_result.cumulative_variance)
    ax.bar(range(1, n + 1), pca_result.explained_variance_ratio, alpha=0.6, label="Individual")
    ax.plot(range(1, n + 1), pca_result.cumulative_variance, "r-o", ms=4, label="Cumulative")
    ax.axhline(0.95, color="gray", linestyle="--", label="95% threshold")
    ax.axvline(pca_result.cev_dim, color="orange", linestyle="--",
               label=f"CEV dim = {pca_result.cev_dim}")

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, n + 0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_msp_attractor(
    components: list,
    component_names: list[str] | None = None,
    n_iterations: int = 80000,
    n_warmup: int = 1000,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the MSP attractor for each component in barycentric simplex coordinates.

    Args:
        components: list of Mess3HMM instances
        component_names: optional display names; defaults to "0", "1", ...
        n_iterations: number of IFS iterations per component
        n_warmup: warmup iterations to discard
        save_path: optional file path to save the figure

    Returns:
        fig: matplotlib Figure
    """
    K = len(components)
    if component_names is None:
        component_names = [str(k) for k in range(K)]

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))
    if K == 1:
        axes = [axes]

    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])

    for ax, comp, name in zip(axes, components, component_names):
        attractor = compute_msp_attractor(comp, n_iterations=n_iterations, n_warmup=n_warmup)
        xy = simplex_to_2d(attractor)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="gray", alpha=0.4, rasterized=True)
        ax.plot(v[:, 0], v[:, 1], "k-", lw=0.8, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        alpha_val = getattr(comp, "alpha", "?")
        x_val = getattr(comp, "x", "?")
        ax.set_title(f"Component {name}  (α={alpha_val}, x={x_val})", fontsize=11)

    fig.suptitle("MSP Attractor — Ground Truth (simplex coordinates)", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_joint_9state_attractor(
    components: list,
    component_names: list[str] | None = None,
    n_iterations: int = 80000,
    n_warmup: int = 1000,
    seed: int = 0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Composite visualization of the joint 9-state belief attractor.

    The joint belief q_{k,s} = π_k × η_{k,s} is non-ergodic: it consists of K
    disconnected MSP fractals, one per component. A naive PCA of the 9D vector is
    dominated by between-component variation (which component?), hiding the fractal
    structure. Instead we show a composite layout:

      - Centre: π-simplex showing the meta-belief space (vertices = pure components)
      - Insets at each vertex: the MSP attractor for that component (η_k in its 2-simplex)

    This captures the full structure: π tells you which corner you're in; within
    each corner the belief traces that component's MSP fractal.

    Args:
        components: list of Mess3HMM instances
        component_names: optional display names
        n_iterations: IFS iterations per component
        n_warmup: warmup iterations to discard
        seed: random seed
        save_path: optional file path to save

    Returns:
        fig: matplotlib Figure
    """
    K = len(components)
    if component_names is None:
        component_names = [str(k) for k in range(K)]

    # Vertices of equilateral triangle for the π-simplex
    pi_verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    pi_v_closed = np.vstack([pi_verts, pi_verts[0]])

    # Inset positions (one per vertex, slightly outside the triangle)
    offsets = [[-0.22, -0.18], [1.02, -0.18], [0.38, np.sqrt(3) / 2 + 0.08]]
    inset_size = 0.28  # fraction of figure width/height

    fig = plt.figure(figsize=(10, 9))

    # ── Centre axis: π-simplex ──────────────────────────────────────────────
    ax_pi = fig.add_axes([0.15, 0.15, 0.70, 0.70])
    ax_pi.plot(pi_v_closed[:, 0], pi_v_closed[:, 1], 'k-', lw=1.5, alpha=0.5)
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for k in range(K):
        ax_pi.scatter(*pi_verts[k], s=120, c=colors[k], zorder=5)
        lbl_offset = [[-0.08, -0.06], [0.02, -0.06], [-0.03, 0.04]][k]
        ax_pi.text(pi_verts[k, 0] + lbl_offset[0], pi_verts[k, 1] + lbl_offset[1],
                   f'Component {component_names[k]}', fontsize=10, color=colors[k], fontweight='bold')
    ax_pi.set_xlim(-0.1, 1.1); ax_pi.set_ylim(-0.1, 1.0)
    ax_pi.set_aspect('equal'); ax_pi.axis('off')
    ax_pi.set_title(
        'Joint 9-state belief: non-ergodic mixture of 3 MSP fractals\n'
        'π (meta-belief) selects which fractal; η_k traces that fractal',
        fontsize=10,
    )

    # Draw arrows from centre to each vertex
    centre = pi_verts.mean(axis=0)
    for k in range(K):
        dx, dy = pi_verts[k] - centre
        ax_pi.annotate('', xy=pi_verts[k] * 0.85 + centre * 0.15,
                        xytext=centre,
                        arrowprops=dict(arrowstyle='->', color=colors[k], lw=1.2, alpha=0.5))

    # ── Inset axes: one MSP fractal per vertex ──────────────────────────────
    inset_positions = [
        [0.01, 0.02, 0.30, 0.30],   # bottom-left  (component A)
        [0.69, 0.02, 0.30, 0.30],   # bottom-right (component B)
        [0.35, 0.65, 0.30, 0.30],   # top-centre   (component C)
    ]
    v_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])

    for k, (comp, pos) in enumerate(zip(components, inset_positions)):
        ax_in = fig.add_axes(pos)
        att = compute_msp_attractor(comp, n_iterations=n_iterations, n_warmup=n_warmup,
                                    seed=seed + k)
        xy = simplex_to_2d(att)
        ax_in.scatter(xy[:, 0], xy[:, 1], s=0.3, c=colors[k], alpha=0.4, rasterized=True)
        ax_in.plot(v_tri[:, 0], v_tri[:, 1], 'k-', lw=0.8, alpha=0.4)
        ax_in.set_aspect('equal'); ax_in.set_xticks([]); ax_in.set_yticks([])
        p = getattr(comp, 'params', comp)
        alpha_val = getattr(comp, 'alpha', '?')
        x_val = getattr(comp, 'x', '?')
        ax_in.set_title(f'η_{component_names[k]}  (α={alpha_val}, x={x_val})', fontsize=8,
                        color=colors[k])
        for spine in ax_in.spines.values():
            spine.set_edgecolor(colors[k]); spine.set_linewidth(1.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def pca_by_layer(
    all_activations: dict[str, np.ndarray],
    n_layers: int,
    cev_threshold: float = 0.95,
) -> list[PCAResult]:
    """
    Run PCA on residual stream activations at each layer.

    Args:
        all_activations: dict mapping "layer_{i}_resid_post" → (N, L, d_model)
        n_layers: number of layers
        cev_threshold: threshold for CEV dimensionality

    Returns:
        pca_results: list of PCAResult, one per layer
    """
    results = []
    for i in range(n_layers):
        acts = all_activations[f"layer_{i}_resid_post"]  # (N, L, d_model)
        N, L, d = acts.shape
        acts_flat = acts.reshape(-1, d)  # (N*L, d_model)
        result = compute_pca(acts_flat, cev_threshold=cev_threshold)
        results.append(result)
    return results
