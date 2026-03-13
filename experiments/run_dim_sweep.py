"""
Dimension sweep experiment: vary d_model ∈ {2, 4, 6, 8} to test whether
the inductive bias toward factored representations survives capacity pressure.

The factored geometry (H2) requires ~8 dimensions minimum:
    2D for π  +  3 × 2D for η_k  =  8 orthogonal dimensions

At d=2 the model must choose between encoding π or η_k — not both.
At d=8 it has exactly enough capacity for the full factored representation.

Usage:
    python experiments/run_dim_sweep.py [--phase PHASE] [--device DEVICE]

Phases:
    train   — Train models at each d_model
    analyze — Analyze all trained models + existing d=64 baseline
    all     — Train then analyze (default)

Output (no existing files overwritten):
    checkpoints/dim_sweep/d{d}/checkpoint_step_020000.pt
    figures/dim_sweep/*.png
    results/dim_sweep_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math
import time

from src.data.mess3 import COMPONENT_PARAMS
from src.data.dataset import Mess3Dataset
from src.model.transformer import build_model
from src.training.train import load_checkpoint, compute_loss
from src.analysis.regression import (
    extract_residual_stream_all_layers,
    fit_belief_regression,
    fit_belief_regression_oos,
)
from src.analysis.orthogonality import full_orthogonality_analysis
from src.analysis.pca import pca_by_layer
from torch.utils.data import DataLoader


# ─── Configuration ───────────────────────────────────────────────────────────

# Each entry: (d_model, n_heads, d_head, d_mlp)
# Constraint: n_heads × d_head == d_model throughout.
DIM_CONFIGS = [
    {"d_model": 2,  "n_heads": 1, "d_head": 2,  "d_mlp": 8},
    {"d_model": 4,  "n_heads": 2, "d_head": 2,  "d_mlp": 16},
    {"d_model": 6,  "n_heads": 2, "d_head": 3,  "d_mlp": 24},
    {"d_model": 8,  "n_heads": 4, "d_head": 2,  "d_mlp": 32},
]

SHARED = {
    "n_layers":    2,
    "n_ctx":       16,
    "d_vocab":     3,
    "act_fn":      "gelu",
    "seed":        42,
    "total_steps": 20000,
    "batch_size":  256,
    "seq_length":  16,
    "lr_max":      1e-3,
    "lr_min":      1e-5,
    "warmup_steps": 1000,
    "n_analysis":  5000,
}

DIRS = {
    "data":           project_root / "data",
    "checkpoints_sweep": project_root / "checkpoints" / "dim_sweep",
    "checkpoints_base":  project_root / "checkpoints",
    "figures_sweep":  project_root / "figures" / "dim_sweep",
    "results":        project_root / "results",
}

COMPONENT_COLORS = ["steelblue", "darkorange", "forestgreen"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_device(override=None):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_val_data(n=None):
    data_path = DIRS["data"]
    val_tokens = torch.from_numpy(np.load(data_path / "val_tokens.npy"))
    val_pi     = np.load(data_path / "val_pi.npy")
    val_eta    = np.load(data_path / "val_eta.npy")
    val_ids    = np.load(data_path / "val_ids.npy")
    if n is not None:
        val_tokens = val_tokens[:n]
        val_pi     = val_pi[:n]
        val_eta    = val_eta[:n]
        val_ids    = val_ids[:n]
    return val_tokens, val_pi, val_eta, val_ids


def final_checkpoint_path(d_model: int) -> Path:
    return DIRS["checkpoints_sweep"] / f"d{d_model}" / f"checkpoint_step_{SHARED['total_steps']:06d}.pt"


def baseline_checkpoint_path() -> Path:
    """Path to the existing d=64 final checkpoint."""
    ckpt_dir = DIRS["checkpoints_base"]
    candidates = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not candidates:
        raise FileNotFoundError("No d=64 baseline checkpoint found in checkpoints/")
    return candidates[-1]


# ─── Training loop (DataLoader-based, no online HMM generation) ──────────────

def _train_from_loader(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    total_steps: int,
    lr_max: float,
    lr_min: float,
    warmup_steps: int,
    checkpoint_every: int,
    log_every: int,
    ckpt_dir: Path,
    device: str,
) -> dict:
    """
    Training loop using pre-generated DataLoader batches.

    Avoids per-step Python HMM sequence generation, which is the bottleneck
    when model forward passes are cheap (tiny d_model).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_max, weight_decay=0.01, betas=(0.9, 0.95)
    )

    history = {"train_loss": [], "val_loss": [], "steps": [], "lr": []}
    step = 0
    t_start = time.time()
    train_iter = iter(train_loader)

    while step < total_steps:
        model.train()

        # Cycle through DataLoader infinitely
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens = batch["tokens"].to(device)

        # Cosine LR with warmup
        if step < warmup_steps:
            lr = lr_max * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss, _ = compute_loss(model, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % log_every == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for i, vbatch in enumerate(val_loader):
                    if i >= 20:
                        break
                    vtokens = vbatch["tokens"].to(device)
                    vl, _ = compute_loss(model, vtokens)
                    val_loss += float(vl.item())
                    n_val += 1
            val_loss /= max(1, n_val)

            elapsed = time.time() - t_start
            print(f"  Step {step:6d}/{total_steps} | train={loss.item():.4f} | "
                  f"val={val_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s")
            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss))
            history["steps"].append(step)
            history["lr"].append(lr)

        if step % checkpoint_every == 0 or step == total_steps:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }
            path = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(ckpt, path)
            print(f"  Saved: {path.name}")

    return history


# ─── Phase: Train ────────────────────────────────────────────────────────────

def phase_train(device: str):
    print("\n" + "=" * 60)
    print("PHASE: Training dim sweep models")
    print("=" * 60)

    # Use pre-saved training tokens — avoids per-step HMM Python generation
    data_path = DIRS["data"]
    train_tokens_np = np.load(data_path / "train_tokens.npy")
    train_pi_np     = np.load(data_path / "train_pi.npy")
    train_eta_np    = np.load(data_path / "train_eta.npy")
    train_ids_np    = np.load(data_path / "train_ids.npy")

    train_dataset = Mess3Dataset(
        tokens=torch.from_numpy(train_tokens_np),
        pi_targets=torch.from_numpy(train_pi_np),
        eta_targets=torch.from_numpy(train_eta_np),
        component_ids=torch.from_numpy(train_ids_np),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=SHARED["batch_size"], shuffle=True, num_workers=0
    )

    val_tokens, val_pi, val_eta, val_ids = load_val_data()
    val_dataset = Mess3Dataset(
        tokens=val_tokens,
        pi_targets=torch.from_numpy(val_pi),
        eta_targets=torch.from_numpy(val_eta),
        component_ids=torch.from_numpy(val_ids),
    )
    val_loader = DataLoader(val_dataset, batch_size=SHARED["batch_size"],
                            shuffle=False, num_workers=0)

    for cfg in DIM_CONFIGS:
        d = cfg["d_model"]
        ckpt_path = final_checkpoint_path(d)

        if ckpt_path.exists():
            print(f"\n[d={d}] Checkpoint already exists — skipping.")
            continue

        print(f"\n{'─'*50}")
        print(f"[d={d}] n_heads={cfg['n_heads']} d_head={cfg['d_head']} d_mlp={cfg['d_mlp']}")
        print(f"{'─'*50}")

        model = build_model(
            n_layers=SHARED["n_layers"],
            d_model=d,
            n_heads=cfg["n_heads"],
            d_head=cfg["d_head"],
            d_mlp=cfg["d_mlp"],
            n_ctx=SHARED["n_ctx"],
            d_vocab=SHARED["d_vocab"],
            act_fn=SHARED["act_fn"],
            seed=SHARED["seed"],
            device=device,
        )

        ckpt_dir = DIRS["checkpoints_sweep"] / f"d{d}"
        _train_from_loader(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            total_steps=SHARED["total_steps"],
            lr_max=SHARED["lr_max"],
            lr_min=SHARED["lr_min"],
            warmup_steps=SHARED["warmup_steps"],
            checkpoint_every=5000,
            log_every=1000,
            ckpt_dir=ckpt_dir,
            device=device,
        )
        print(f"[d={d}] Done.")


# ─── Phase: Analyze ──────────────────────────────────────────────────────────

def _analyze_one(model, val_tokens, val_pi, val_eta, val_ids, d_model, device):
    """
    Run regression, orthogonality, and PCA analysis for one model.
    Returns a dict of scalar metrics.
    """
    K = val_pi.shape[2]
    n_layers = model.cfg.n_layers
    last_layer = n_layers - 1

    # Extract residual stream
    all_acts = extract_residual_stream_all_layers(model, val_tokens, device=device)
    acts_final = all_acts[f"layer_{last_layer}_resid_post"][:, -1, :]  # (N, d)
    pi_final   = val_pi[:, -1, :]      # (N, K)
    eta_final  = val_eta[:, -1, :, :]  # (N, K, 3)

    # R² (OOS) for π — n_subspace_components capped at d_model
    n_pi_comp = min(K - 1, d_model)
    _, pi_r2_oos = fit_belief_regression_oos(
        acts_final, pi_final, n_subspace_components=n_pi_comp
    )

    # R² (OOS) per η_k
    eta_r2_oos = []
    for k in range(K):
        n_eta_comp = min(2, d_model)
        _, r2k = fit_belief_regression_oos(
            acts_final, eta_final[:, k, :], n_subspace_components=n_eta_comp
        )
        eta_r2_oos.append(float(r2k))

    # Subspace overlap (mean off-diagonal) — note: uninformative when d <= 4
    # because all subspaces span the full space
    ortho = full_orthogonality_analysis(acts_final, pi_final, eta_final, K)
    mat = ortho["overlap_matrix"]
    n = mat.shape[0]
    off_diag = mat[np.triu_indices(n, k=1)]
    mean_overlap = float(off_diag.mean())

    # PCA CEV dimensionality
    pca_results = pca_by_layer(all_acts, n_layers)
    cev_dim = pca_results[last_layer].cev_dim

    return {
        "d_model":      d_model,
        "pi_r2_oos":    float(pi_r2_oos),
        "eta_r2_oos":   eta_r2_oos,
        "mean_overlap": mean_overlap,
        "cev_dim":      cev_dim,
        "acts_final":   acts_final,   # kept for scatter plots
        "pi_final":     pi_final,
        "eta_final":    eta_final,
        "val_ids":      val_ids,
    }


def phase_analyze(device: str):
    print("\n" + "=" * 60)
    print("PHASE: Analyzing all models")
    print("=" * 60)

    DIRS["figures_sweep"].mkdir(parents=True, exist_ok=True)
    DIRS["results"].mkdir(parents=True, exist_ok=True)

    val_tokens, val_pi, val_eta, val_ids = load_val_data(n=SHARED["n_analysis"])
    K = val_pi.shape[2]

    all_results = []

    # ── Sweep models (d=2,4,6,8) ──────────────────────────────────────────
    for cfg in DIM_CONFIGS:
        d = cfg["d_model"]
        ckpt = final_checkpoint_path(d)
        if not ckpt.exists():
            print(f"[d={d}] No checkpoint found — skipping. Run --phase train first.")
            continue

        print(f"\n[d={d}] Loading checkpoint...")
        model = build_model(
            n_layers=SHARED["n_layers"],
            d_model=d,
            n_heads=cfg["n_heads"],
            d_head=cfg["d_head"],
            d_mlp=cfg["d_mlp"],
            n_ctx=SHARED["n_ctx"],
            d_vocab=SHARED["d_vocab"],
            act_fn=SHARED["act_fn"],
            seed=SHARED["seed"],
            device=device,
        )
        model, step, history = load_checkpoint(model, str(ckpt), device)
        model.eval()

        val_loss_final = history["val_loss"][-1] if history.get("val_loss") else float("nan")
        print(f"[d={d}] Step={step}  val_loss={val_loss_final:.4f}")

        res = _analyze_one(model, val_tokens, val_pi, val_eta, val_ids, d, device)
        res["val_loss"] = val_loss_final
        all_results.append(res)

        print(f"  π R² OOS      = {res['pi_r2_oos']:.3f}")
        for k, r2k in enumerate(res["eta_r2_oos"]):
            print(f"  η_{k} R² OOS  = {r2k:.3f}")
        print(f"  mean overlap  = {res['mean_overlap']:.3f}  {'(note: uninformative at d≤4)' if d <= 4 else ''}")
        print(f"  CEV dim       = {res['cev_dim']}")

    # ── Baseline d=64 ─────────────────────────────────────────────────────
    try:
        base_ckpt = baseline_checkpoint_path()
        print(f"\n[d=64] Loading baseline from {base_ckpt.name}...")
        model64 = build_model(
            n_layers=2, d_model=64, n_heads=4, d_head=16, d_mlp=256,
            n_ctx=16, d_vocab=3, seed=SHARED["seed"], device=device,
        )
        model64, step64, hist64 = load_checkpoint(model64, str(base_ckpt), device)
        model64.eval()
        val_loss64 = hist64["val_loss"][-1] if hist64.get("val_loss") else float("nan")
        print(f"[d=64] Step={step64}  val_loss={val_loss64:.4f}")

        res64 = _analyze_one(model64, val_tokens, val_pi, val_eta, val_ids, 64, device)
        res64["val_loss"] = val_loss64
        all_results.append(res64)

        print(f"  π R² OOS      = {res64['pi_r2_oos']:.3f}")
        for k, r2k in enumerate(res64["eta_r2_oos"]):
            print(f"  η_{k} R² OOS  = {r2k:.3f}")
        print(f"  mean overlap  = {res64['mean_overlap']:.3f}")
        print(f"  CEV dim       = {res64['cev_dim']}")

    except FileNotFoundError as e:
        print(f"[d=64] {e}")

    if not all_results:
        print("No results to plot. Run --phase train first.")
        return

    # Sort by d_model
    all_results.sort(key=lambda r: r["d_model"])

    _plot_results(all_results, K)
    _save_summary(all_results)


def _plot_results(all_results: list[dict], K: int):
    fig_path = DIRS["figures_sweep"]
    dims = [r["d_model"] for r in all_results]
    component_names = [p.name for p in COMPONENT_PARAMS]

    # ── Plot 1: R² vs d_model ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    pi_r2s = [r["pi_r2_oos"] for r in all_results]
    ax.plot(dims, pi_r2s, "ko-", lw=2, ms=8, label="π (meta-belief)", zorder=5)

    for k in range(K):
        eta_r2s = [r["eta_r2_oos"][k] for r in all_results]
        ax.plot(dims, eta_r2s, "o--", color=COMPONENT_COLORS[k], lw=1.5, ms=6,
                label=f"η_{component_names[k]} (within-component {k})")

    ax.axvline(8, color="gray", linestyle=":", lw=1.5, label="d=8 (minimum for full factored)")
    ax.set_xlabel("d_model", fontsize=12)
    ax.set_ylabel("R² (out-of-sample)", fontsize=12)
    ax.set_title("Belief State Encoding vs Model Dimension", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(dims)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path / "r2_vs_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path / 'r2_vs_dim.png'}")

    # ── Plot 2: Subspace overlap vs d_model ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    overlaps = [r["mean_overlap"] for r in all_results]
    bars = ax.bar(dims, overlaps, color="steelblue", alpha=0.7, width=0.8)
    ax.axvline(8, color="gray", linestyle=":", lw=1.5, label="d=8 minimum")
    ax.set_xlabel("d_model", fontsize=12)
    ax.set_ylabel("Mean off-diagonal subspace overlap", fontsize=12)
    ax.set_title("Subspace Orthogonality vs Model Dimension\n"
                 "(note: overlap is inflated at d≤4 — subspaces span the full space)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(dims)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, overlaps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path / "overlap_vs_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path / 'overlap_vs_dim.png'}")

    # ── Plot 3: Val loss vs d_model ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    losses = [r["val_loss"] for r in all_results]
    ax.plot(dims, losses, "rs-", lw=2, ms=8)
    ax.axvline(8, color="gray", linestyle=":", lw=1.5, label="d=8 minimum")
    ax.set_xlabel("d_model", fontsize=12)
    ax.set_ylabel("Final validation loss", fontsize=12)
    ax.set_title("Task Performance vs Model Dimension", fontsize=13)
    ax.set_xticks(dims)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for x, y in zip(dims, losses):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path / "loss_vs_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path / 'loss_vs_dim.png'}")

    # ── Plot 4: 2D scatter for d=2 ────────────────────────────────────────
    # At d=2 the residual stream is already 2D — no PCA needed.
    # We colour by (a) component identity and (b) within-component state to see
    # which the model has chosen to represent.
    small_d_results = [r for r in all_results if r["d_model"] <= 8]
    for res in small_d_results:
        d = res["d_model"]
        acts = res["acts_final"]     # (N, d)
        ids  = res["val_ids"]        # (N,)
        eta  = res["eta_final"]      # (N, K, 3)
        K_   = eta.shape[1]

        # If d > 2, use first 2 PCA directions
        if d == 2:
            xy = acts
            x_label, y_label = "Residual dim 0", "Residual dim 1"
        else:
            from sklearn.decomposition import PCA as _PCA
            pca2 = _PCA(n_components=2).fit(acts)
            xy = pca2.transform(acts)
            x_label, y_label = "PC 1", "PC 2"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: colour by component identity (π)
        for k in range(K_):
            mask = ids == k
            axes[0].scatter(xy[mask, 0], xy[mask, 1],
                            s=4, c=COMPONENT_COLORS[k], alpha=0.4,
                            label=f"Component {component_names[k]}", rasterized=True)
        axes[0].set_title(f"d={d}: Coloured by component identity (π)", fontsize=11)
        axes[0].set_xlabel(x_label); axes[0].set_ylabel(y_label)
        axes[0].legend(fontsize=9, markerscale=3)

        # Right: colour by dominant within-component state (η of true component)
        # For each point, get η_k for its true component, find dominant state
        dominant_states = np.array([
            int(np.argmax(eta[i, ids[i], :]))
            for i in range(len(ids))
        ])
        state_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        for s in range(3):
            mask = dominant_states == s
            axes[1].scatter(xy[mask, 0], xy[mask, 1],
                            s=4, c=state_colors[s], alpha=0.4,
                            label=f"State {s} dominant", rasterized=True)
        axes[1].set_title(f"d={d}: Coloured by dominant within-component state (η_k)", fontsize=11)
        axes[1].set_xlabel(x_label); axes[1].set_ylabel(y_label)
        axes[1].legend(fontsize=9, markerscale=3)

        fig.suptitle(
            f"d={d} Residual Stream — Which Belief State Is Encoded?\n"
            f"Cleaner clusters on the left → π encoded | right → η_k encoded",
            fontsize=11,
        )
        fig.tight_layout()
        fname = fig_path / f"scatter_d{d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


def _save_summary(all_results: list[dict]):
    summary = []
    for r in all_results:
        summary.append({
            "d_model":      r["d_model"],
            "val_loss":     r["val_loss"],
            "pi_r2_oos":    r["pi_r2_oos"],
            "eta_r2_oos":   r["eta_r2_oos"],
            "mean_overlap": r["mean_overlap"],
            "cev_dim":      r["cev_dim"],
        })

    out_path = DIRS["results"] / "dim_sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dimension sweep experiment")
    parser.add_argument("--phase", choices=["train", "analyze", "all"], default="all")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.phase in ("train", "all"):
        phase_train(device)

    if args.phase in ("analyze", "all"):
        phase_analyze(device)

    print("\n" + "=" * 60)
    print("Dim sweep complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
