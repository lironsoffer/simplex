"""
Training loop for the Mess3 transformer experiment.

Uses cross-entropy next-token prediction loss.
Saves checkpoints every checkpoint_every steps.
Logs per-step and per-position loss.
"""

import os
import json
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from src.data.mess3 import Mess3HMM


def sample_batch_online(
    components: list[Mess3HMM],
    batch_size: int,
    seq_length: int,
    rng: np.random.Generator,
    device: str,
) -> torch.Tensor:
    """
    Generate a fresh batch of token sequences on-the-fly from the HMM components.

    Each sequence is generated entirely from one randomly chosen component.
    This prevents overfitting to a fixed dataset.

    Args:
        components: list of Mess3HMM instances
        batch_size: number of sequences in the batch
        seq_length: length of each sequence
        rng: numpy random generator (stateful, advances each call)
        device: torch device

    Returns:
        tokens: (batch_size, seq_length) int64 tensor on device
    """
    K = len(components)
    tokens_np = np.zeros((batch_size, seq_length), dtype=np.int64)

    for i in range(batch_size):
        k = rng.integers(0, K)
        tokens_np[i], _ = components[k].generate_sequence(seq_length, rng=rng)

    return torch.from_numpy(tokens_np).to(device)


def compute_loss(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross-entropy next-token prediction loss.

    Args:
        model: HookedTransformer
        tokens: (batch, seq_len) int64

    Returns:
        loss: scalar, mean over all positions
        per_position_loss: (seq_len - 1,) mean loss at each position
    """
    logits = model(tokens)  # (batch, seq_len, d_vocab)

    # Shift: predict token t+1 from position t
    logits_shifted = logits[:, :-1, :]   # (batch, L-1, d_vocab)
    targets = tokens[:, 1:]               # (batch, L-1)

    # Loss per position
    batch_size, L_minus_1, d_vocab = logits_shifted.shape
    per_token_loss = F.cross_entropy(
        logits_shifted.reshape(-1, d_vocab),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, L_minus_1)  # (batch, L-1)

    per_position_loss = per_token_loss.mean(dim=0)  # (L-1,)
    loss = per_position_loss.mean()

    return loss, per_position_loss.detach()


def get_cosine_lr(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
    warmup_steps: int = 1000,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def train(
    model: HookedTransformer,
    val_loader: DataLoader,
    components: list[Mess3HMM],
    total_steps: int = 50000,
    batch_size: int = 256,
    seq_length: int = 16,
    lr_max: float = 1e-3,
    lr_min: float = 1e-5,
    warmup_steps: int = 1000,
    checkpoint_every: int = 5000,
    early_checkpoint_steps: list[int] | None = None,
    log_every: int = 200,
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
    seed: int = 0,
    # Kept for backward compat but ignored (online generation is always used)
    train_loader: DataLoader | None = None,
) -> dict:
    """
    Train the transformer on next-token prediction using on-the-fly data generation.

    Sequences are generated fresh each step from the HMM components, preventing
    overfitting to a fixed dataset. The HMMs can generate infinite unique sequences.

    Args:
        model: HookedTransformer
        val_loader: validation DataLoader (fixed, for consistent evaluation)
        components: list of Mess3HMM instances for on-the-fly generation
        total_steps: total training steps
        batch_size: number of sequences per step
        seq_length: token sequence length
        lr_max: peak learning rate
        lr_min: minimum learning rate at end of schedule
        warmup_steps: linear warmup steps
        checkpoint_every: save checkpoint every N steps
        early_checkpoint_steps: additional steps at which to save checkpoints (e.g. [100, 500, 1000, 2000])
        log_every: log metrics every N steps
        checkpoint_dir: directory to save checkpoints
        device: torch device (auto-detected if None)
        seed: random seed for sequence generation
        train_loader: ignored (kept for backward compatibility)

    Returns:
        history: dict with training metrics
    """
    if device is None:
        device = str(next(model.parameters()).device)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_max,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "per_position_loss": [],
        "steps": [],
        "lr": [],
    }

    _early_steps = set(early_checkpoint_steps or [])

    step = 0
    rng = np.random.default_rng(seed)
    t_start = time.time()

    while step < total_steps:
        model.train()

        # Generate fresh batch on-the-fly (no overfitting)
        tokens = sample_batch_online(components, batch_size, seq_length, rng, device)

        # Update learning rate
        lr = get_cosine_lr(step, total_steps, lr_max, lr_min, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        optimizer.zero_grad()
        loss, _ = compute_loss(model, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Logging
        if step % log_every == 0 or step == total_steps:
            val_loss, val_per_pos = _eval(model, val_loader, device, max_batches=20)

            elapsed = time.time() - t_start
            print(
                f"Step {step:6d}/{total_steps} | "
                f"train_loss={loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"time={elapsed:.0f}s"
            )

            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss))
            history["per_position_loss"].append(val_per_pos.tolist())
            history["steps"].append(step)
            history["lr"].append(lr)

        # Checkpointing (regular interval + early steps for phase-transition capture)
        if step % checkpoint_every == 0 or step == total_steps or step in _early_steps:
            _save_checkpoint(model, optimizer, step, history, checkpoint_path)

    # Save final history
    with open(checkpoint_path / "history.json", "w") as f:
        json.dump(history, f)

    return history


def _eval(
    model: HookedTransformer,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, np.ndarray]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    per_pos_accum = None
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break
            tokens = batch["tokens"].to(device)
            loss, per_pos = compute_loss(model, tokens)
            total_loss += float(loss.item())
            if per_pos_accum is None:
                per_pos_accum = per_pos.cpu().numpy()
            else:
                per_pos_accum += per_pos.cpu().numpy()
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_per_pos = per_pos_accum / max(1, n_batches) if per_pos_accum is not None else np.array([])
    return avg_loss, avg_per_pos


def _save_checkpoint(
    model: HookedTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    history: dict,
    checkpoint_path: Path,
):
    """Save model checkpoint."""
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    path = checkpoint_path / f"checkpoint_step_{step:06d}.pt"
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    model: HookedTransformer,
    checkpoint_path: str,
    device: str | None = None,
) -> tuple[HookedTransformer, int, dict]:
    """
    Load a model checkpoint.

    Args:
        model: HookedTransformer (must match architecture of checkpoint)
        checkpoint_path: path to .pt checkpoint file
        device: target device

    Returns:
        model: model with loaded weights
        step: training step of checkpoint
        history: training history up to that step
    """
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if device:
        model = model.to(device)
    return model, ckpt["step"], ckpt.get("history", {})
