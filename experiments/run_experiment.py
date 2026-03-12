"""
End-to-end experiment runner for the non-ergodic Mess3 transformer experiment.

Usage:
    python experiments/run_experiment.py [--phase PHASE] [--device DEVICE]

Phases:
    data    — Generate and save dataset
    train   — Train transformer (requires data)
    analyze — Run all geometry analyses (requires trained model)
    all     — Run all phases in sequence (default)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.mess3 import (
    build_default_components,
    compute_synchronisation_horizon,
    COMPONENT_PARAMS,
)
from src.data.belief_update import initial_belief
from src.data.dataset import generate_sequences, build_dataloaders, Mess3Dataset
from src.model.transformer import build_model, get_residual_stream
from src.training.train import train, load_checkpoint
from src.analysis.regression import (
    extract_residual_stream_all_layers,
    fit_belief_regression,
    fit_belief_regression_oos,
    fit_component_subspace,
    r2_by_layer_and_position,
    project_out_subspace,
)
from src.analysis.pca import (
    compute_pca,
    pca_by_layer,
    plot_fractal,
    plot_joint_9state_attractor,
    plot_pca_explained_variance,
    compute_msp_attractor,
    simplex_to_2d,
)
from src.analysis.orthogonality import (
    full_orthogonality_analysis,
    vary_one_analysis,
    pairwise_overlap_matrix,
)
from src.analysis.context_dynamics import (
    r2_vs_position_all_layers,
    compute_nstar_analytical,
    plot_r2_vs_position,
    dimensionality_vs_position,
)
from src.analysis.training_dynamics import (
    run_training_dynamics,
    plot_training_dynamics,
    find_phase_transition,
)


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    "n_train": 50000,
    "n_val": 5000,
    "seq_length": 16,
    "batch_size": 256,
    "total_steps": 50000,
    "lr_max": 1e-3,
    "lr_min": 1e-5,
    "warmup_steps": 1000,
    "checkpoint_every": 5000,
    "log_every": 200,
    "n_analysis_sequences": 5000,
    "seed": 42,
    # Model
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "d_head": 16,
    "d_mlp": 256,
    "n_ctx": 16,
    "d_vocab": 3,
}

DIRS = {
    "data": project_root / "data",
    "checkpoints": project_root / "checkpoints",
    "figures": project_root / "figures",
    "results": project_root / "results",
}


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Phase 1: Data Generation ───────────────────────────────────────────────

def phase_data():
    print("\n" + "=" * 60)
    print("PHASE 1: Data Generation")
    print("=" * 60)

    ensure_dirs()
    components = build_default_components()

    # Print component parameters and synchronisation horizon
    print("\nComponent parameters:")
    for i, (comp, params) in enumerate(zip(components, COMPONENT_PARAMS)):
        print(f"  Component {params.name}: α={params.alpha}, x={params.x}")
        print(f"    Stationary: {comp.stationary_distribution.round(4)}")

    n_star = compute_nstar_analytical(components)
    print(f"\nSynchronisation horizon N* ≈ {n_star:.2f} (context window = {CONFIG['seq_length']})")
    if n_star >= CONFIG["seq_length"]:
        print("  WARNING: N* >= context window! Components may not be distinguishable.")

    # Generate datasets
    print(f"\nGenerating {CONFIG['n_train']} training sequences...")
    train_tokens, train_pi, train_eta, train_ids = generate_sequences(
        CONFIG["n_train"], CONFIG["seq_length"], components, seed=CONFIG["seed"]
    )

    print(f"Generating {CONFIG['n_val']} validation sequences...")
    val_tokens, val_pi, val_eta, val_ids = generate_sequences(
        CONFIG["n_val"], CONFIG["seq_length"], components, seed=CONFIG["seed"] + 1
    )

    # Sanity checks
    print("\nSanity checks:")
    print(f"  Belief state sums (should be 1): pi={train_pi.sum(axis=-1).mean():.6f}, "
          f"eta={train_eta.sum(axis=-1).mean():.6f}")
    comp_counts = np.bincount(train_ids, minlength=3)
    print(f"  Component distribution: {comp_counts} ({comp_counts / len(train_ids) * 100}%)")

    # Save datasets
    data_path = DIRS["data"]
    np.save(data_path / "train_tokens.npy", train_tokens)
    np.save(data_path / "train_pi.npy", train_pi)
    np.save(data_path / "train_eta.npy", train_eta)
    np.save(data_path / "train_ids.npy", train_ids)
    np.save(data_path / "val_tokens.npy", val_tokens)
    np.save(data_path / "val_pi.npy", val_pi)
    np.save(data_path / "val_eta.npy", val_eta)
    np.save(data_path / "val_ids.npy", val_ids)

    # Save config
    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "n_star": float(n_star),
        "component_params": [
            {"name": p.name, "alpha": p.alpha, "x": p.x} for p in COMPONENT_PARAMS
        ],
    }
    with open(data_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to {data_path}")
    return components, n_star


# ─── Phase 2: Training ──────────────────────────────────────────────────────

def phase_train(device: str):
    print("\n" + "=" * 60)
    print("PHASE 2: Training")
    print("=" * 60)

    ensure_dirs()
    data_path = DIRS["data"]

    # Load validation data (fixed set for consistent eval)
    val_tokens = torch.from_numpy(np.load(data_path / "val_tokens.npy"))
    val_pi = np.load(data_path / "val_pi.npy")
    val_eta = np.load(data_path / "val_eta.npy")
    val_ids = np.load(data_path / "val_ids.npy")

    from src.data.dataset import Mess3Dataset
    from torch.utils.data import DataLoader

    val_dataset = Mess3Dataset(
        tokens=val_tokens,
        pi_targets=torch.from_numpy(val_pi),
        eta_targets=torch.from_numpy(val_eta),
        component_ids=torch.from_numpy(val_ids),
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0)

    # Build model
    model = build_model(
        n_layers=CONFIG["n_layers"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        d_head=CONFIG["d_head"],
        d_mlp=CONFIG["d_mlp"],
        n_ctx=CONFIG["n_ctx"],
        d_vocab=CONFIG["d_vocab"],
        seed=CONFIG["seed"],
        device=device,
    )

    # Train with online data generation (prevents memorization of fixed dataset)
    # Early checkpoints at steps 100, 500, 1000, 2000 capture phase-transition dynamics
    components = build_default_components()
    history = train(
        model=model,
        val_loader=val_loader,
        components=components,
        total_steps=CONFIG["total_steps"],
        batch_size=CONFIG["batch_size"],
        seq_length=CONFIG["seq_length"],
        lr_max=CONFIG["lr_max"],
        lr_min=CONFIG["lr_min"],
        warmup_steps=CONFIG["warmup_steps"],
        checkpoint_every=CONFIG["checkpoint_every"],
        early_checkpoint_steps=[100, 500, 1000, 2000],
        log_every=CONFIG["log_every"],
        checkpoint_dir=str(DIRS["checkpoints"]),
        device=device,
        seed=CONFIG["seed"],
    )

    print(f"\nTraining complete. Final val_loss={history['val_loss'][-1]:.4f}")
    return model, history


# ─── Phase 3: Geometry Analysis ─────────────────────────────────────────────

def phase_analyze(device: str):
    print("\n" + "=" * 60)
    print("PHASE 3: Geometry Analysis")
    print("=" * 60)

    ensure_dirs()
    data_path = DIRS["data"]
    fig_path = DIRS["figures"]
    results_path = DIRS["results"]

    components = build_default_components()
    n_star = compute_nstar_analytical(components)
    component_names = [p.name for p in COMPONENT_PARAMS]

    # Load a subset of validation data for analysis
    N_analysis = CONFIG["n_analysis_sequences"]
    val_tokens = torch.from_numpy(np.load(data_path / "val_tokens.npy")[:N_analysis])
    val_pi = np.load(data_path / "val_pi.npy")[:N_analysis]
    val_eta = np.load(data_path / "val_eta.npy")[:N_analysis]
    val_ids = np.load(data_path / "val_ids.npy")[:N_analysis]
    K = val_pi.shape[2]

    # Load final checkpoint
    from src.analysis.training_dynamics import get_checkpoint_paths
    ckpt_paths = get_checkpoint_paths(str(DIRS["checkpoints"]))
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoints found. Run training first.")
    final_step, final_ckpt = ckpt_paths[-1]

    model = build_model(
        n_layers=CONFIG["n_layers"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        d_head=CONFIG["d_head"],
        d_mlp=CONFIG["d_mlp"],
        n_ctx=CONFIG["n_ctx"],
        d_vocab=CONFIG["d_vocab"],
        seed=CONFIG["seed"],
        device=device,
    )
    model, step, _ = load_checkpoint(model, final_ckpt, device)
    model.eval()
    print(f"Loaded checkpoint at step {step}")

    last_layer = CONFIG["n_layers"] - 1

    # ── 3a. PCA Dimensionality ──
    print("\n[3a] PCA Dimensionality...")
    all_acts = extract_residual_stream_all_layers(model, val_tokens, device=device)
    pca_results = pca_by_layer(all_acts, CONFIG["n_layers"])

    fig, axes = plt.subplots(1, CONFIG["n_layers"], figsize=(6 * CONFIG["n_layers"], 4))
    if CONFIG["n_layers"] == 1:
        axes = [axes]
    for i, pca_res in enumerate(pca_results):
        plot_pca_explained_variance(
            pca_res,
            title=f"Layer {i} PCA (CEV dim={pca_res.cev_dim})",
            ax=axes[i],
        )
    fig.suptitle("PCA Explained Variance by Layer", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path / "pca_explained_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pca_dims = [r.cev_dim for r in pca_results]
    print(f"  CEV dimensionality by layer: {pca_dims}")

    # ── 3b. R² Regression ──
    print("\n[3b] R² by Layer and Position...")
    r2_results = r2_vs_position_all_layers(
        model, val_tokens, val_pi, val_eta, device=device
    )

    # Plot R² heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, data) in zip(axes, [
        ("π meta-belief R² (in-sample)", r2_results["pi_r2"]),
        (f"η_0 R² Component A (in-sample)", r2_results["eta_r2"][:, :, 0]),
    ]):
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis",
                       origin="lower", extent=[1, CONFIG["seq_length"], 0, CONFIG["n_layers"]])
        ax.set_xlabel("Context position")
        ax.set_ylabel("Layer")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(fig_path / "r2_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # OOS R² at final layer, final position (avoiding position-1 near-zero-variance artifact)
    print("\n  Out-of-sample R² at final layer, final position (position 1 excluded — near-uniform targets):")
    acts_final_for_oos = all_acts[f"layer_{last_layer}_resid_post"][:, -1, :]
    pi_final_for_oos = val_pi[:, -1, :]
    _, pi_r2_oos = fit_belief_regression_oos(acts_final_for_oos, pi_final_for_oos,
                                              n_subspace_components=K - 1)
    print(f"    π R² OOS = {pi_r2_oos:.3f}")
    eta_r2_oos_list = []
    for k in range(K):
        eta_k_final = val_eta[:, -1, k, :]
        _, eta_k_r2_oos = fit_belief_regression_oos(acts_final_for_oos, eta_k_final,
                                                     n_subspace_components=2)
        eta_r2_oos_list.append(float(eta_k_r2_oos))
        print(f"    η_{k} R² OOS = {eta_k_r2_oos:.3f}")

    # ── 3c. Context Dynamics ──
    print("\n[3c] Context Dynamics...")
    positions = np.arange(CONFIG["seq_length"])

    fig_ctx = plot_r2_vs_position(
        positions=positions,
        pi_r2=r2_results["pi_r2"][last_layer],
        eta_r2=r2_results["eta_r2"][last_layer],
        n_star=n_star,
        component_names=component_names,
        title=f"R² vs Context Position (Layer {last_layer})",
    )
    fig_ctx.savefig(fig_path / "r2_vs_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ctx)

    # ── 3d. Orthogonality Analysis ──
    print("\n[3d] Orthogonality Analysis...")
    # Use final position (most context) for orthogonality analysis
    acts_final = all_acts[f"layer_{last_layer}_resid_post"][:, -1, :]
    pi_final = val_pi[:, -1, :]
    eta_final = val_eta[:, -1, :, :]

    ortho_results = full_orthogonality_analysis(
        activations=acts_final,
        pi_targets=pi_final,
        eta_targets=eta_final,
        K=K,
    )

    print(f"  π R² (in-sample)={ortho_results['pi_r2']:.3f}")
    for k in range(K):
        print(f"  η_{k} R² (in-sample)={ortho_results['eta_r2'][k]:.3f}")
    print(f"\n  Subspace overlap matrix:")
    names_row = "    " + " " * 8
    for name_j in ortho_results["overlap_names"]:
        names_row += name_j.ljust(8)
    print(names_row)
    for i, name_i in enumerate(ortho_results["overlap_names"]):
        row = "    " + name_i.ljust(8)
        for j in range(len(ortho_results["overlap_names"])):
            row += f"  {ortho_results['overlap_matrix'][i, j]:.3f}  "
        print(row)

    print("\n  Projection tests (remove π, check η_k R² drop):")
    for k, pt in enumerate(ortho_results["projection_pi_into_eta"]):
        print(f"    η_{k}: before={pt.r2_before:.3f}, after={pt.r2_after:.3f}, drop={pt.r2_drop:.4f}")

    print("\n  Projection tests (remove η_k, check π R² drop):")
    for k, pt in enumerate(ortho_results["projection_eta_into_pi"]):
        print(f"    η_{k} removed: π before={pt.r2_before:.3f}, after={pt.r2_after:.3f}, drop={pt.r2_drop:.4f}")

    print("\n  Cross-projection matrix [j,k] = R² drop in η_k after removing η_j subspace:")
    cross_mat = ortho_results["cross_projection_matrix"]
    header = "    " + " " * 6
    for k in range(K):
        header += f"  η_{k}   "
    print(header)
    for j in range(K):
        row = f"    η_{j} removed: "
        for k in range(K):
            if j == k:
                row += "   --   "
            else:
                row += f"  {cross_mat[j, k]:.4f}"
        print(row)

    # ── 3e. MSP Fractals (coloured, one panel per component) ──
    print("\n[3e] MSP Fractals...")
    from src.analysis.pca import compute_msp_attractor, simplex_to_2d
    colors_msp = ["steelblue", "darkorange", "forestgreen"]
    fig_msp, axes_msp = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes_msp = [axes_msp]
    for k, (comp, ax, color) in enumerate(zip(components, axes_msp, colors_msp)):
        attractor = compute_msp_attractor(comp, n_iterations=80000)
        xy = simplex_to_2d(attractor)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c=color, alpha=0.4)
        ax.set_title(f"Component {component_names[k]}\nα={comp.alpha}, x={comp.x}")
        ax.set_aspect("equal")
        ax.set_xlabel("Simplex dim 1")
        ax.set_ylabel("Simplex dim 2")
    fig_msp.suptitle("Ground-Truth MSP Fractals for Each Mess3 Component", y=1.02)
    fig_msp.tight_layout()
    fig_msp.savefig(fig_path / "msp_fractals.png", dpi=150, bbox_inches="tight")
    plt.close(fig_msp)

    # ── 3e2. Joint 9-State Attractor ──
    print("\n[3e2] Joint 9-State Attractor...")
    fig_joint = plot_joint_9state_attractor(
        components=components,
        component_names=component_names,
        save_path=str(fig_path / "joint_9state_attractor.png"),
    )
    plt.close(fig_joint)

    # ── 3f. Fractal Visualisation ──
    print("\n[3f] Fractal Visualisation...")
    # Use all post-synchronisation positions (n_star_pos to ctx-1) rather than only
    # the last position, to densely sample the fractal. Each component gets
    # n_sequences_k × n_post_positions data points.
    n_star_pos = min(int(np.ceil(n_star)), CONFIG["seq_length"] - 1)
    n_post = CONFIG["seq_length"] - n_star_pos
    print(f"  Using positions {n_star_pos}–{CONFIG['seq_length']-1} ({n_post} positions after N*≈{n_star:.1f})")

    fig_frac, axes_frac = plt.subplots(K, 2, figsize=(10, 5 * K))
    if K == 1:
        axes_frac = axes_frac[np.newaxis, :]

    for k in range(K):
        mask = val_ids == k
        if mask.sum() < 10:
            print(f"  Component {k}: insufficient sequences ({mask.sum()}), skipping fractal")
            continue

        # Stack all post-N* positions for component k sequences
        acts_k = all_acts[f"layer_{last_layer}_resid_post"][mask, n_star_pos:, :]  # (N_k, n_post, d)
        eta_k  = val_eta[mask, n_star_pos:, k, :]                                  # (N_k, n_post, 3)
        acts_k_flat = acts_k.reshape(-1, CONFIG["d_model"])   # (N_k*n_post, d)
        eta_k_flat  = eta_k.reshape(-1, 3)                    # (N_k*n_post, 3)

        print(f"  Component {component_names[k]}: {mask.sum()} seqs × {n_post} positions = {len(acts_k_flat)} points")
        plot_fractal(
            activations_k=acts_k_flat,
            eta_k_targets=eta_k_flat,
            component=components[k],
            component_name=component_names[k],
            axes=axes_frac[k],
        )

    fig_frac.suptitle(
        "Fractal Geometry: Ground Truth (left) vs Activations — Regression Output (right)",
        y=1.01,
    )
    fig_frac.tight_layout()
    fig_frac.savefig(fig_path / "fractal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_frac)

    # ── 3f. Vary-One Analysis ──
    print("\n[3g] Vary-One Analysis...")
    # For each component k, collect activations and η_k targets
    vary_one_acts = {}
    vary_one_eta = {}
    for k in range(K):
        mask = val_ids == k
        if mask.sum() > 0:
            acts_k = all_acts[f"layer_{last_layer}_resid_post"][mask, -1, :]
            eta_k = val_eta[mask, -1, k, :]
            vary_one_acts[k] = acts_k
            vary_one_eta[k] = eta_k

    if len(vary_one_acts) == K:
        vary_one_results = vary_one_analysis(
            activations_by_component=vary_one_acts,
            eta_targets_by_component=vary_one_eta,
            K=K,
        )
        print("  Per-component η_k R²:")
        for k in range(K):
            print(f"    Component {component_names[k]}: {vary_one_results['per_component_r2'][k]:.3f}")
        print("  Cross-component subspace overlap:")
        for i in range(K):
            row = f"    Component {component_names[i]}: "
            for j in range(K):
                row += f"{vary_one_results['cross_component_overlap'][i, j]:.3f}  "
            print(row)

    # ── 3g. Training Dynamics ──
    print("\n[3h] Training Dynamics...")
    dynamics = run_training_dynamics(
        checkpoint_dir=str(DIRS["checkpoints"]),
        tokens=val_tokens,
        pi_targets=val_pi,
        eta_targets=val_eta,
        model_kwargs={
            "n_layers": CONFIG["n_layers"],
            "d_model": CONFIG["d_model"],
            "n_heads": CONFIG["n_heads"],
            "d_head": CONFIG["d_head"],
            "d_mlp": CONFIG["d_mlp"],
            "n_ctx": CONFIG["n_ctx"],
            "d_vocab": CONFIG["d_vocab"],
            "seed": CONFIG["seed"],
        },
        layer=-1,
        position=-1,
        device=device,
    )

    fig_dyn = plot_training_dynamics(
        dynamics,
        component_names=component_names,
        title="Belief Subspace Emergence over Training",
    )
    fig_dyn.savefig(fig_path / "training_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig_dyn)

    pi_transition = find_phase_transition(dynamics["steps"], dynamics["pi_r2"], threshold=0.9)
    eta_transitions = [
        find_phase_transition(dynamics["steps"], dynamics["eta_r2"][:, k], threshold=0.9)
        for k in range(K)
    ]
    print(f"  π phase transition (R²>0.9) at step: {pi_transition}")
    for k in range(K):
        print(f"  η_{k} phase transition at step: {eta_transitions[k]}")

    # ── Save all results ──
    summary = {
        "n_star": float(n_star),
        "pca_dims_by_layer": pca_dims,
        "pi_r2_final_insample": float(ortho_results["pi_r2"]),
        "eta_r2_final_insample": [float(r) for r in ortho_results["eta_r2"]],
        "pi_r2_final_oos": float(pi_r2_oos),
        "eta_r2_final_oos": eta_r2_oos_list,
        "overlap_matrix": ortho_results["overlap_matrix"].tolist(),
        "overlap_names": ortho_results["overlap_names"],
        "projection_remove_pi_check_eta": [
            {"r2_before": pt.r2_before, "r2_after": pt.r2_after, "r2_drop": pt.r2_drop}
            for pt in ortho_results["projection_pi_into_eta"]
        ],
        "projection_remove_eta_check_pi": [
            {"r2_before": pt.r2_before, "r2_after": pt.r2_after, "r2_drop": pt.r2_drop}
            for pt in ortho_results["projection_eta_into_pi"]
        ],
        "cross_projection_matrix": ortho_results["cross_projection_matrix"].tolist(),
        "pi_r2_vs_position": r2_results["pi_r2"].tolist(),
        "eta_r2_vs_position": r2_results["eta_r2"].tolist(),
        "training_dynamics": {
            "steps": dynamics["steps"].tolist(),
            "pi_r2": dynamics["pi_r2"].tolist(),
            "eta_r2": dynamics["eta_r2"].tolist(),
            "pi_transition_step": pi_transition,
            "eta_transition_steps": eta_transitions,
        },
    }

    with open(results_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Figures saved to {fig_path}")
    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mess3 Transformer Experiment")
    parser.add_argument(
        "--phase",
        choices=["data", "train", "analyze", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (auto-detected if not specified)",
    )
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")

    if args.phase in ("data", "all"):
        phase_data()

    if args.phase in ("train", "all"):
        phase_train(device)

    if args.phase in ("analyze", "all"):
        phase_analyze(device)

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
