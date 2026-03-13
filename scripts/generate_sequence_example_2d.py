"""
Generate a 2D grid figure showing one example sequence per Mess3 component.
Rows = components (A, B, C); columns = token positions (0–15).
Cell color encodes token value; dashed line marks the synchronisation horizon N*.

Run from the project root:
    python scripts/generate_sequence_example_2d.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.mess3 import Mess3HMM, Mess3Params

# ---------------------------------------------------------------------------
# Component parameters
# ---------------------------------------------------------------------------
COMPONENTS = [
    (Mess3Params(alpha=0.95, x=0.02), "Component A"),
    (Mess3Params(alpha=0.80, x=0.08), "Component B"),
    (Mess3Params(alpha=0.55, x=0.25), "Component C"),
]
SEEDS = [454, 100, 200]
SEQ_LEN = 16
N_STAR = 9.6

# Token colors consistent with the rest of the paper
TOKEN_COLORS = ["steelblue", "darkorange", "forestgreen"]

# ---------------------------------------------------------------------------
# Generate sequences
# ---------------------------------------------------------------------------
sequences = []
for (params, _), seed in zip(COMPONENTS, SEEDS):
    rng = np.random.default_rng(seed)
    hmm = Mess3HMM(params)
    tokens, _ = hmm.generate_sequence(SEQ_LEN, rng)
    sequences.append(tokens)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(-0.5, SEQ_LEN - 0.5)
ax.set_ylim(-0.5, len(COMPONENTS) + 0.2)  # extra headroom above position labels
ax.set_aspect("equal")

for row, (tokens, (_, name)) in enumerate(zip(sequences, COMPONENTS)):
    y = len(COMPONENTS) - 1 - row  # flip so Component A is on top
    for col, tok in enumerate(tokens):
        rect = plt.Rectangle(
            (col - 0.45, y - 0.4), 0.9, 0.8,
            facecolor=TOKEN_COLORS[tok],
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(col, y, str(tok), ha="center", va="center",
                color="white", fontsize=13, fontweight="bold")
    # Row label
    ax.text(-0.7, y, name, ha="right", va="center",
            fontsize=11, fontweight="bold")

# Column (position) labels
for col in range(SEQ_LEN):
    ax.text(col, len(COMPONENTS) - 0.3, str(col),
            ha="center", va="center", fontsize=9, color="gray")
ax.text(-0.7, len(COMPONENTS) - 0.3, "t =",
        ha="right", va="center", fontsize=9, color="gray")

# N* dashed line
ax.axvline(x=N_STAR, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(N_STAR + 0.15, len(COMPONENTS) - 0.15,
        r"$N^* \approx 9.6$", fontsize=10, va="center")

ax.set_title("Example sequences from each component", fontsize=13, pad=15)
ax.axis("off")

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "sequence_example_2d.png"
)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
