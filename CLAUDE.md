# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Workflow

Claude acts as **team lead**. For any non-trivial task, spawn specialized subagents rather than doing everything inline:

- Use `Agent` tool with `run_in_background=true` for independent parallel tasks (data gen, training, analysis, plot generation)
- Use `Agent` tool in foreground when you need the result before proceeding
- Delegate: file edits, test runs, plot generation, code search to subagents
- Keep the main context for orchestration, decisions, and communicating results to the user

## Original Task

> Please implement the following experiment, and submit a PDF writeup of the results, alongside the code. It may be helpful to refer to our publications for definitions of processes and how to derive belief state geometry, in particular, Transformers Learn Factored Representations and the references therein. For computational feasibility, we recommend making the neural networks small (e.g., 1–3 layers, context window 8–16).
>
> Construct a non-ergodic training dataset, consisting of Mess3 processes with different parameters, where each training sequence is generated entirely by one Mess3 ergodic component. Train a transformer on this data via next-token prediction. Why do you think this type of structure is interesting and/or relevant to language models?
>
> Make a pre-registered prediction (honor code) as to what geometry the activations should take, and how it should change with context position and across layers. Derive this prediction mathematically as far as you can, and separately give your intuition for what the geometry should look like. You will not be penalized for getting this wrong—we are interested in both your formal reasoning and your geometric intuition. Are there multiple possible geometries you can think of?
>
> After training, analyze the residual stream geometry. What structure is there? How does it relate to the belief geometries of the component processes?
>
> Perform at least one additional analysis of your choosing that you think is interesting or informative. Tell us why you chose it. If you don't have time to implement it, describe what you would do.

**Deliverables:** (1) code, (2) PDF writeup covering motivation, pre-registered predictions (mathematical + intuitive), geometry analysis results, and at least one additional analysis with justification.

## Project Overview

This is a mechanistic interpretability research project. The central question: does a transformer trained on a non-ergodic mixture of K Mess3 HMM processes learn a **factored representation** of the joint belief state — separating meta-belief π (which component?) from within-component beliefs η_k (where within that component?) into orthogonal residual stream subspaces?

The project is in early implementation phase. Read `project_plan.md` for the full experimental design and `theoretical_background.md` for the math.

## Setup (to be implemented)

**Dependencies to install:**

- PyTorch
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — required for residual stream access
- NumPy, scikit-learn (PCA, linear regression), matplotlib

**Reference repos to clone before coding:**

- `adamimos/epsilon-transformers` — HMM data generation (Mess3), transformer training, regression analysis pipeline. Start here.
- `Astera-org/factored-reps` — vary-one analysis, CEV dimensionality, subspace orthogonality metrics. Port these tools.

## Architecture

**Model:** Small transformer, 2 layers, model dim 64, context window 16, trained on next-token prediction over alphabet {0, 1, 2}. Use TransformerLens for interpretability access.

**Data:** K=3 Mess3 HMMs with parameters (each component is individually ergodic; non-ergodicity arises at the corpus level because sequences never switch between components):


| Component | α    | x    |
| --------- | ---- | ---- |
| A         | 0.85 | 0.05 |
| B         | 0.70 | 0.15 |
| C         | 0.55 | 0.25 |


Each training sequence is generated entirely by one sampled component (non-ergodic). Ground-truth belief states `(π(t), η₁(t), η₂(t), η₃(t))` are computed analytically using the two-step update rule and stored as regression targets.

## Implementation Steps

See `project_plan.md` for full detail. The five phases are:

1. **Data generation** — implement Mess3 HMMs with token-labeled transition matrices, generate sequences + ground-truth belief states
2. **Training** — train transformer, log per-token loss, save checkpoints
3. **Geometry analysis** — PCA dimensionality, linear regression (residual → π and η_k), orthogonality checks, projection tests, fractal visualisation
4. **Context dynamics** — R² vs context position, look for sharp increase around N*
5. **Training dynamics** — repeat geometry analysis at each checkpoint

## Key Math

**Two-step belief update** (implement this for ground-truth generation):

- Step 1 — meta-belief: `π'_k = π_k · L_k(x) / Z` where `L_k(x) = η_k · T_k^(x) · 1`
- Step 2 — within-component: `η_k' = η_k · T_k^(x) / L_k(x)`

**Synchronisation horizon:** `N* ≈ 1 / min_{k≠k'} D_KL(P_k || P_k')` — verify N* < 16 for chosen parameters.

**Competing hypotheses to test:**

- H1 (joint): π and η_k share an undifferentiated subspace — projecting out π degrades η_k regression
- H2 (factored): K+1 orthogonal subspaces (2D π subspace + three 2D η_k subspaces); projecting out π leaves η_k R² ≈ 1
- H3 (superposition): π and η_k recoverable approximately but with non-orthogonal interference

## Key References

### Shai et al. NeurIPS 2024 — "Transformers represent belief state geometry in their residual stream" ([2405.15943](https://arxiv.org/abs/2405.15943))

**Setup:** Transformers trained on synthetic HMM processes (including Mess3) with known latent structure.

**Key finding:** Belief states η(t) are linearly encoded in the residual stream. The set of activations, projected onto a 2D subspace, traces the Mixed State Presentation (MSP) fractal — the attractor of the belief state IFS. Transformers learn representations extending beyond the local next-token prediction objective.

**Methods introduced:** Regression-based subspace identification (fit linear map from residual stream to ground-truth belief states, extract top singular vectors as subspace basis). This is the core analysis method for this project.

---

### Riechers et al. 2025 — "Next-token pretraining implies in-context learning" ([2505.18373](https://arxiv.org/abs/2505.18373))

**Setup:** Information-theoretic framework applied to non-ergodic training corpora (mixtures of sources); validated on synthetic datasets.

**Key finding:** ICL emerges necessarily from non-ergodic pretraining — a model's in-context performance is mathematically coupled to the ensemble of tasks in pretraining. Non-ergodic sources produce power-law (not exponential) decay of in-context loss. Predicts phase transitions during induction head formation.

**Methods introduced:** Synchronisation horizon `N* ≈ 1 / min D_KL(P_k || P_k')` — the context length at which the model commits to one component. Architecture- and modality-independent information-theoretic framework.

---

### Shai et al. ICML 2025 — "Transformers learn factored representations" ([2602.02385](https://arxiv.org/abs/2602.02385))

**Setup:** Transformers trained on parallel joint processes (multiple simultaneously-observed conditionally-independent sources, e.g. 3 Mess3 + 2 Bloch Walk).

**Key finding:** Models have an inductive bias toward factored representations when factors are conditionally independent — learning orthogonal subspaces (linear dimensional growth) rather than dense product-space representations (exponential growth). This preference persists even when conditional independence breaks down.

**Methods introduced:** Competing hypotheses framework (H1 joint / H2 factored / H3 superposition); vary-one analysis; cumulative explained variance (CEV) dimensionality measure; subspace overlap metric (Appendix H). **Port these tools for this project.**


