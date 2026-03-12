# Project Plan: Belief State Geometry in Non-Ergodic Transformers

## Research Question

Does a transformer trained on a non-ergodic mixture of K Mess3 processes learn a **factored representation** of the joint belief state — separating the meta-belief π (which component am I in?) from the within-component beliefs η_k (where within that component?) into orthogonal subspaces of the residual stream?

---

## Setup

**Data:** K=3 Mess3 processes with different parameters, one process per training sequence.

Suggested parameters:

| Component | α    | x    | Character                        |
|-----------|------|------|----------------------------------|
| A         | 0.85 | 0.05 | Sharp fractal, slow mixing       |
| B         | 0.70 | 0.15 | Medium fractal, moderate mixing  |
| C         | 0.55 | 0.25 | Soft fractal, faster mixing      |

Choose parameters so that N* = 1 / min D_KL(P_k || P_k') < context window length.

**Token alphabet:** {0, 1, 2} (shared across all components — same alphabet, different emission/transition parameters).

**Architecture:** Small transformer (2 layers, model dim 64, context window 16) trained via next-token prediction. Use TransformerLens for interpretability access.

**Ground truth:** For every token position in every sequence, compute the exact belief state (π(t), η₁(t), η₂(t), η₃(t)) analytically using the two-step update rule. This is the regression target.

---

## Competing Hypotheses

### H1 — Joint Representation
The residual stream represents η_joint directly as one undifferentiated (3K-1 = 8)-dimensional subspace. The π and η_k structure exists mathematically but is not reflected in the geometric organisation of the residual stream. Removing directions associated with π also removes information needed for η_k.

### H2 — Factored Representation *(predicted)*
The residual stream carves its 8 dimensions into K+1 orthogonal subspaces:
- One 2D subspace encoding π (the meta-belief simplex)
- Three separate 2D subspaces, one per η_k (each containing that component's Mess3 fractal)

Predicted by the block-diagonal structure of the joint HMM and the conditional independence of components given component identity.

### H3 — Superposition
The model encodes more features than dimensions by using non-orthogonal near-orthogonal directions. π and η_k features are all present but entangled, recoverable approximately but with interference between subspaces.

---

## Pre-Registered Predictions (H2)

1. **Dimensionality:** PCA of residual stream (final layer) shows exactly 8 non-negligible principal components, structured as 4 groups of 2.

2. **Fractal geometry:** Projecting activations onto the η_k subspace for component k reveals that component's Mess3 fractal (shape determined by its (α, x) parameters). The three component fractals are geometrically distinguishable.

3. **Orthogonality:** Cosine similarity between the π subspace directions and each η_k subspace direction ≈ 0.

4. **Linear decodability:** Linear regression from residual stream to ground-truth π achieves R² ≈ 1. Same for each η_k separately.

5. **Orthogonal decodability test:** After projecting out the π subspace from the residual stream, linear regression to η_k still achieves R² ≈ 1 (and vice versa). This is the key test distinguishing H2 from H1.

6. **Context dynamics:** R² for π increases sharply around position N* as the model commits to one component. R² for η_k is high throughout but the dominant component's η_k becomes cleanest past N*.

7. **Layer dynamics:** π subspace decodable from earlier layers; η_k fractal structure cleaner in later layers.

---

## Implementation Steps

### Step 1 — Data Generation
- Implement K=3 Mess3 HMMs with their token-labeled transition matrices T_k^(x)
- Generate training sequences: for each sequence, sample component k ~ Uniform(K), then run that Mess3 for L=16 tokens
- Generate ground-truth belief states: for each sequence, run the two-step update rule at every position to get (π(t), η₁(t), η₂(t), η₃(t))
- Split into train/val sets

### Step 2 — Training
- Train small transformer on next-token prediction cross-entropy
- Log per-token loss over training steps, per position
- Check for loss "bump" (induction head phase transition) and power-law vs exponential decay of loss with context position
- Save checkpoints at multiple training stages

### Step 3 — Geometry Analysis
For each layer, each context position, each checkpoint:

**3a. Dimensionality**
- PCA on residual stream activations
- Plot cumulative explained variance — how many components needed for 95%?
- Compare: full dataset vs sequences from one component only

**3b. Regression**
- Fit linear regression: residual stream activations → ground-truth π(t)
- Fit linear regression: residual stream activations → ground-truth η_k(t) for each k
- Report R² for each, broken down by layer and context position

**3c. Orthogonality**
- Extract subspace directions from each regression (top singular vectors)
- Compute pairwise cosine similarities between π subspace and each η_k subspace
- Compute subspace overlap metric (from factored representations paper Appendix H)

**3d. Projection test**
- Project π subspace out of residual stream
- Re-run η_k regression on residual — does R² hold?
- Project each η_k subspace out — does π regression hold?

**3e. Fractal visualisation**
- For sequences from component k where π has collapsed (t >> N*), project residual stream onto η_k subspace and plot in 2D
- Compare to ground-truth MSP fractal for that component's (α, x) parameters

### Step 4 — Context Dynamics
- Plot R²(π, t) vs context position — look for sharp increase at N*
- Plot R²(η_k, t) vs context position — look for which component's fractal is cleanest
- Compute effective dimensionality of residual stream vs context position

### Step 5 — Training Dynamics
- Repeat geometry analysis at each saved checkpoint
- Track: when does π subspace crystallise? When does η_k fractal emerge?
- Connection to ICLR two-phase dynamics: does π (concept formation) precede η_k (architectural reorganisation)?

---

## Analysis Tools (from factored representations paper)

- **Vary-one analysis:** hold K-1 components fixed, vary one — activations should move only within that component's subspace
- **Cumulative explained variance (CEV):** dimensionality measure from PCA
- **Subspace overlap metric:** measures angle between subspaces, defined in Appendix H of 2602.02385
- **Regression-based subspace identification:** fit linear map, extract top singular vectors as subspace basis

---

## Key Reference Papers

| Paper | arXiv | What it contributes |
|---|---|---|
| Shai et al. NeurIPS 2024 — "Transformers represent belief state geometry in their residual stream" | [arXiv:2405.15943](https://arxiv.org/abs/2405.15943) | Single-component baseline: fractal in residual stream, regression method |
| Riechers et al. 2025 — "Next-token pretraining implies in-context learning" | [arXiv:2505.18373](https://arxiv.org/abs/2505.18373) | Non-ergodic ICL: power-law scaling, induction heads, two-level belief structure |
| Shai et al. ICML 2025 — "Transformers learn factored representations" | [arXiv:2602.02385](https://arxiv.org/abs/2602.02385) | Factored representations: analysis methods, orthogonality metrics, competing hypotheses |

## Repositories

| Repo | Description | Priority |
|---|---|---|
| [adamimos/epsilon-transformers](https://github.com/adamimos/epsilon-transformers/tree/main) | Official repo by Adam Shai. HMM data generation (Mess3, RRXOR), transformer training, analysis notebooks. Uses TransformerLens. | **Start here** — has the real HMM generation code and regression analysis pipeline |
| [Astera-org/factored-reps](https://github.com/Astera-org/factored-reps/tree/master) | Official repo for the factored representations paper. Contains the vary-one analysis, CEV dimensionality measurement, and subspace orthogonality metrics. | **Key for analysis** — port these analysis tools to your temporal mixture setup |
| [sanowl/BeliefStateTransformer](https://github.com/sanowl/BeliefStateTransformer/tree/main) | Community reimplementation with extensions: layer-wise analysis, training dynamics tracking. | **Optional reference** — useful if epsilon-transformers is hard to navigate |

---

## What Makes This Novel

The three reference papers cover:
- Single ergodic Mess3 → fractal geometry (NeurIPS 2024)
- K processes in temporal mixture → ICL loss dynamics (ICL paper)
- K processes in parallel → factored geometry (factored reps paper)

This project covers the missing cell: **K processes in temporal mixture → factored geometry**. The factored representations paper's analysis tools apply, but its experimental setup is different. The ICL paper's training setup matches, but it doesn't look at residual stream geometry. This is a genuine gap.
