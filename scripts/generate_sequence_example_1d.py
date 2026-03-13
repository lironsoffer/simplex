"""
Generate a clean illustration of a sample Mess3 sequence with belief state annotations.
Suitable for a NeurIPS-style paper figure.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data ---
tokens = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2]
pi_a   = [0.333, 0.333, 0.483, 0.609, 0.705, 0.777, 0.831, 0.871,
          0.903, 0.927, 0.945, 0.959, 0.969, 0.895, 0.912, 0.937]
n_tokens = len(tokens)

positions = list(range(1, n_tokens + 1))

# --- Colors ---
TOKEN_COLORS = {
    0: '#4682B4',   # steelblue
    1: '#FF8C00',   # darkorange
    2: '#228B22',   # forestgreen
}
TOKEN_NAMES = {0: 'Token 0', 1: 'Token 1', 2: 'Token 2'}

# --- Figure & axes setup ---
# We use a data coordinate system where each token occupies 1 unit on x.
# x-centers: 1, 2, ..., 16
# y: we place rows at fixed data-y values

fig, ax = plt.subplots(figsize=(14, 3.6))

# Leave left margin for row labels, right margin for last box
LEFT_MARGIN  = 0.8   # data units reserved for row labels
RIGHT_MARGIN = 0.6
X_START = LEFT_MARGIN         # x-center of token 1
X_STEP  = 1.0                 # spacing between centers

BOX_W = 0.78   # box width  (data units)
BOX_H = 0.40   # box height (data units) — keep it squarish in display

# y-coordinates (data)
Y_TOP    = 1.00   # top of figure data range
Y_POS    = 0.88   # position number row
Y_BOX_C  = 0.60   # center of token boxes
Y_PI     = 0.30   # π_A value row
Y_BOTTOM = 0.10   # bottom annotation space

ax.set_xlim(0.0, X_START + (n_tokens - 1) * X_STEP + RIGHT_MARGIN + 0.5)
ax.set_ylim(Y_BOTTOM - 0.05, Y_TOP + 0.08)

# Remove all spines and ticks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# --- Row labels on the left (in data coords) ---
label_x = LEFT_MARGIN - 0.55
ax.text(label_x, Y_BOX_C, 'Token:', va='center', ha='right',
        fontsize=10, color='#333333', fontweight='bold')
ax.text(label_x, Y_PI, r'$\pi_A$:', va='center', ha='right',
        fontsize=10, color='#555555', fontstyle='italic')

# --- Draw each token box ---
for i, (tok, pi, pos) in enumerate(zip(tokens, pi_a, positions)):
    x = X_START + i * X_STEP

    color = TOKEN_COLORS[tok]

    # Square token box
    rect = mpatches.FancyBboxPatch(
        (x - BOX_W / 2, Y_BOX_C - BOX_H / 2),
        BOX_W, BOX_H,
        boxstyle='round,pad=0.03',
        linewidth=1.0,
        edgecolor='white',
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)

    # Token digit inside box
    ax.text(x, Y_BOX_C, str(tok), va='center', ha='center',
            fontsize=12, color='white', fontweight='bold', zorder=4)

    # Position number above box
    ax.text(x, Y_POS, str(pos), va='center', ha='center',
            fontsize=8, color='#777777')

    # π_A value below box
    ax.text(x, Y_PI, f'{pi:.3f}', va='center', ha='center',
            fontsize=8.5, color='#444444')

# --- N* vertical dashed line (between positions 9 and 10) ---
nstar_x = X_START + 8.5 * X_STEP   # between index 8 (pos 9) and 9 (pos 10)
ax.axvline(x=nstar_x, ymin=0.05, ymax=0.98,
           color='#CC0000', linestyle='--', linewidth=1.4, alpha=0.8, zorder=2)
ax.text(nstar_x + 0.06, Y_TOP + 0.04, r'$N^* \approx 9.6$',
        va='top', ha='left', fontsize=9.5, color='#CC0000')

# --- Annotation: π_A > 0.9 with arrow pointing to position 9 ---
pos9_x = X_START + 8 * X_STEP   # position 9 (index 8)
ax.annotate(
    r'$\pi_A > 0.9$',
    xy=(pos9_x, Y_PI - 0.04),
    xytext=(pos9_x - 2.2, Y_BOTTOM + 0.01),
    fontsize=9.5,
    color='#333333',
    arrowprops=dict(
        arrowstyle='->',
        color='#444444',
        lw=1.1,
        connectionstyle='arc3,rad=-0.2',
    ),
    va='bottom',
    ha='center',
)

# --- Legend ---
legend_patches = [
    mpatches.Patch(facecolor=TOKEN_COLORS[k], edgecolor='#999999',
                   linewidth=0.8, label=TOKEN_NAMES[k])
    for k in sorted(TOKEN_COLORS)
]
ax.legend(handles=legend_patches, loc='upper right',
          fontsize=9, framealpha=0.9, edgecolor='#cccccc',
          handlelength=1.1, handleheight=0.95,
          bbox_to_anchor=(1.0, 1.08))

# --- Title ---
ax.set_title(
    'Example sequence from Component A  (true component shown in blue)',
    fontsize=11, pad=8, color='#222222', loc='left', x=0.0,
)

plt.tight_layout(pad=0.4)
plt.savefig('/Users/lironsoffer/workspace/simplex/figures/sequence_example.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: figures/sequence_example.png')
