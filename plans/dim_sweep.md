# Plan: Dimension Sweep — Factored Geometry Under Capacity Pressure

## Hypothesis

The factored representation (H2) requires a minimum of ~8 dimensions:
- 2D subspace for π (meta-belief over K=3 components → K-1=2 free dimensions)
- 2D subspace per η_k (3-state Mess3 → 2 free dimensions), ×3 components
- Total: 2 + 3×2 = **8 orthogonal dimensions**

At d=64, the model has 8× surplus capacity — factoring is the path of least resistance.
The question: **is the inductive bias toward factoring strong enough to survive capacity pressure?**

Sharp predictions:
- **d=2**: cannot represent both π and any η_k. Model must choose. Prediction: learns π (meta-belief is most predictively valuable — it determines the entire output distribution).
- **d=4**: can fit π (2D) + one η_k (2D). Which η_k gets the slot? Prediction: the most discriminative component (lowest-entropy Mess3).
- **d=6**: can fit π (2D) + two η_k's.
- **d=8**: exactly the minimum for full factored — all subspaces fit with zero slack.

---

## Architecture Per Dimension

All models: 2 layers, n_ctx=16, d_vocab=3, act_fn="gelu".
`d_mlp = 4 × d_model` throughout. `n_heads` chosen so `d_head = d_model / n_heads` is an integer ≥ 1.

| d_model | n_heads | d_head | d_mlp | approx params |
|---------|---------|--------|-------|---------------|
| 2       | 1       | 2      | 8     | ~0.5k         |
| 4       | 2       | 2      | 16    | ~1k           |
| 6       | 2       | 3      | 24    | ~2k           |
| 8       | 4       | 2      | 32    | ~3k           |
| 64      | 4       | 16     | 256   | ~101k (done)  |

---

## File Layout (no conflicts with existing files)

```
checkpoints/
  dim_sweep/
    d2/   checkpoint_step_*.pt, history.json
    d4/   ...
    d6/   ...
    d8/   ...
  checkpoint_step_*.pt   ← existing d=64 checkpoints, untouched

figures/
  dim_sweep/
    r2_vs_dim.png          ← main result: R²(π), R²(η_k) vs d
    subspace_overlap_vs_dim.png
    pca_cev_vs_dim.png
  *.png                   ← existing figures, untouched

results/
  dim_sweep_summary.json  ← new (existing summary.json untouched)
  summary.json            ← existing, untouched

experiments/
  run_dim_sweep.py        ← new runner (run_experiment.py untouched)
```

---

## Implementation Steps

### Step 1 — New runner: `experiments/run_dim_sweep.py`

Single self-contained script. No changes to existing files.

```python
DIM_CONFIGS = [
    {"d_model": 2,  "n_heads": 1, "d_head": 2,  "d_mlp": 8},
    {"d_model": 4,  "n_heads": 2, "d_head": 2,  "d_mlp": 16},
    {"d_model": 6,  "n_heads": 2, "d_head": 3,  "d_mlp": 24},
    {"d_model": 8,  "n_heads": 4, "d_head": 2,  "d_mlp": 32},
]
TOTAL_STEPS = 20000   # sufficient for these tiny models
BATCH_SIZE  = 256
SEQ_LENGTH  = 16
```

For each config:
1. Build model with `build_model(**dim_cfg, n_layers=2, n_ctx=16, d_vocab=3)`
2. Train with `train(..., checkpoint_dir=f"checkpoints/dim_sweep/d{d_model}", total_steps=20000)`
   - Uses the same on-the-fly data generation as the main experiment
   - No need to touch `data/` — reuses the same `val_tokens.npy` for validation
3. Save final checkpoint

### Step 2 — Analysis loop

For each trained model (d ∈ {2, 4, 6, 8}) plus the existing d=64 model:

1. Load final checkpoint
2. Extract residual stream at final layer, final position
3. Run:
   - `fit_belief_regression_oos(acts, pi_targets)` → `pi_r2`
   - `fit_belief_regression_oos(acts, eta_k_targets)` → `eta_r2[k]` for k=0,1,2
   - `full_orthogonality_analysis(acts, pi, eta)` → subspace overlap matrix
   - `pca_by_layer(all_acts)` → CEV dimensionality

All analysis functions already exist in `src/analysis/` — no new analysis code needed.

### Step 3 — Plots

**Plot 1: R² vs d_model** (main result)
- x-axis: d_model ∈ {2, 4, 6, 8, 64}
- y-axis: R²
- Lines: π (solid), η_A (dashed), η_B (dashed), η_C (dashed)
- Vertical dashed line at d=8 (theoretical minimum for full factored)

**Plot 2: Subspace overlap vs d_model**
- For each d, one number: mean off-diagonal subspace overlap
- Shows when orthogonality breaks down

**Plot 3: Val loss vs d_model**
- Baseline: does d=8 actually learn the task as well as d=64?

---

## What to Look For

| Observation | Interpretation |
|-------------|----------------|
| d=2: R²(π) ≈ 1, R²(η_k) ≈ 0 | Model prioritises component identity |
| d=2: R²(π) ≈ 0, R²(η_k) ≈ 1 for one k | Model prioritises within-component tracking |
| d=2: all R² ≈ 0 | Model can't represent belief states at d=2 at all |
| d=8: R²(π) ≈ 1, R²(η_k) ≈ 1, overlap ≈ 0 | Full factored representation, tight |
| d=4,6: R²(η_k) rises one-by-one | Model adds subspaces as capacity permits |
| d=8: overlap > 0 vs d=64: overlap ≈ 0 | Orthogonality degrades under capacity pressure |

The d=2 case is the most theoretically interesting. The 2D simplex for π and the 2D simplex for η_k are geometrically identical. The model must decide which one to embed in its 2D residual stream. This is a direct test of the relative value the transformer places on meta-belief vs within-component belief.

---

## Run Order

```bash
cd /Users/lironsoffer/workspace/simplex
python experiments/run_dim_sweep.py --device mps
```

Expected runtime: ~5–10 min total (models are tiny, 20k steps each).

---

## Diagnostic Caveat: Regression at d=2 and d=4

The standard `fit_belief_regression_oos` fits a linear map from `d_model`-dimensional activations to a 2D belief simplex. At d=2, the activations *are* 2D — the regression has zero degrees of freedom to project. R² will be near 1 trivially for *any* linearly structured representation. High R² at d=2 is therefore uninformative on its own.

**What to do instead at d=2 (and d=4):**

1. **Fit both π and η_k regressions and compare residuals.** The model encodes *something* 2D — the question is which belief state that something corresponds to. Whichever regression has lower residual error wins.

2. **Visual check:** Plot the 2D residual stream activations (no PCA needed — it's already 2D) with two colourings:
   - Colour by component identity (which of A/B/C generated the sequence) → tests whether π is encoded
   - Colour by within-component state bin (discretised η_k) → tests whether η_k is encoded

   Whichever colouring produces cleaner cluster separation answers the question directly.

3. **Mutual information proxy:** Compute I(activations ; π) vs I(activations ; η_k) using a simple k-NN estimator. The higher MI wins.

The visual check is the most interpretable and should be the primary diagnostic for d=2. Add one 2D scatter panel per model to the figure output.

---

## Open Questions

1. At d=2, does the model learn π or η_k? Why?
2. Does R²(η_k) climb in any particular order across k at d=4,6? (Prediction: lowest-entropy component first.)
3. At d=8, is subspace overlap near-zero (clean factored) or elevated (pressure artifacts)?
4. Does val loss at d=8 match d=64, or is there a performance gap?
