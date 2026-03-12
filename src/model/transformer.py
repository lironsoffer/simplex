"""
TransformerLens GPT-2-style model for the Mess3 experiment.

Architecture (CLAUDE.md spec):
    n_layers=2, d_model=64, n_heads=4, d_head=16, d_mlp=256
    n_ctx=16, d_vocab=3, act_fn="gelu"

TransformerLens is used to provide built-in hooks for residual stream access.
"""

import torch
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_model(
    n_layers: int = 2,
    d_model: int = 64,
    n_heads: int = 4,
    d_head: int = 16,
    d_mlp: int = 256,
    n_ctx: int = 16,
    d_vocab: int = 3,
    act_fn: str = "gelu",
    seed: int = 42,
    device: str | None = None,
) -> HookedTransformer:
    """
    Build a small HookedTransformer for next-token prediction on Mess3 data.

    Args:
        n_layers: number of transformer layers
        d_model: model dimension (residual stream width)
        n_heads: number of attention heads
        d_head: dimension per attention head
        d_mlp: MLP hidden dimension
        n_ctx: context window size
        d_vocab: vocabulary size (3 for Mess3 tokens {0,1,2})
        act_fn: MLP activation function
        seed: random seed for weight initialisation
        device: torch device (auto-detected if None)

    Returns:
        model: HookedTransformer ready for training
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch.manual_seed(seed)

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_mlp=d_mlp,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        act_fn=act_fn,
        normalization_type="LN",
        attention_dir="causal",
        attn_only=False,
        tokenizer_name=None,
        seed=seed,
        init_weights=True,
        positional_embedding_type="standard",
    )

    model = HookedTransformer(cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: {n_layers}L d={d_model} heads={n_heads} | {n_params:,} params | device={device}")

    return model


def get_residual_stream(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Extract residual stream activations at every layer via TransformerLens hooks.

    Hook names follow TransformerLens convention:
        "blocks.{layer}.hook_resid_pre"  — before layer layer
        "blocks.{layer}.hook_resid_post" — after layer layer (incl. MLP)
        "hook_embed" — token embedding (pre-positional)
        "hook_pos_embed" — positional embedding

    Args:
        model: HookedTransformer
        tokens: (batch, seq_len) int64 token indices

    Returns:
        cache: dict mapping hook name → activation tensor (batch, seq_len, d_model)
    """
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: (
            "hook_resid_post" in name
            or "hook_resid_pre" in name
            or name in ("hook_embed", "hook_pos_embed")
        ),
        return_type="logits",
    )
    return {k: v.detach() for k, v in cache.items()}


def residual_stream_at_layer(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    hook_type: str = "resid_post",
) -> torch.Tensor:
    """
    Extract residual stream activations at a specific layer.

    Args:
        model: HookedTransformer
        tokens: (batch, seq_len) int64
        layer: layer index (0-indexed)
        hook_type: "resid_pre" or "resid_post"

    Returns:
        activations: (batch, seq_len, d_model)
    """
    hook_name = f"blocks.{layer}.hook_{hook_type}"
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name == hook_name,
        return_type="logits",
    )
    return cache[hook_name].detach()
