# Belief State Geometry in Non-Ergodic Transformers: A Mess3 Experiment

**Author:** [Your Name]
**Date:** March 2026

---

## 1. Motivation

### Why Non-Ergodic Mess3 Training?

Real language corpora are **non-ergodic**: each document belongs to one domain, author, or style — and crucially, it never switches mid-document. A model trained on such data must simultaneously (a) identify *which* type of document it is reading (meta-belief π) and (b) track fine-grained structure within that document type (within-component belief η_k). These two tasks are epistemically distinct but must be solved together.

We formalise this with **K=3 Mess3 HMMs** — 3-state, 3-token edge-emitting Hidden Markov Models — where each training sequence is generated entirely by one component (non-ergodic corpus construction). This yields a clean, analytically tractable proxy for the more complex real-world situation.

**Connection to in-context learning (ICL):** Riechers et al. (2025) prove that ICL emerges *necessarily* from non-ergodic pretraining. The model's in-context performance is coupled to its ability to identify which component is generating the current sequence. The synchronisation horizon N* = 1/KL_min is the number of tokens needed to commit to one component. By choosing parameters with N* < context window length (16), we ensure this commitment happens within a single sequence.

**Connection to factored representations:** Shai et al. (2025) show that transformers trained on parallel joint processes learn orthogonal subspaces in the residual stream — one per factor. Our experiment tests whether the same phenomenon occurs for *temporal* mixtures (one factor per sequence, not simultaneous factors). This is the missing cell in the literature.

**The gap:** Three papers cover adjacent setups:
- Shai et al. 2024: single ergodic Mess3 → fractal geometry in residual stream
- Riechers et al. 2025: non-ergodic mixture → ICL loss dynamics (but not geometry)
- Shai et al. 2025: parallel factors → factored geometry (but different setup)

**This experiment:** non-ergodic mixture → factored geometry.

---

## 2. Setup

### Data

**K=3 Mess3 components:**

| Component | α    | x    | Character |
|-----------|------|------|-----------|
| A         | 0.85 | 0.05 | Sharp fractal, slow mixing |
| B         | 0.70 | 0.15 | Medium fractal, moderate mixing |
| C         | 0.55 | 0.25 | Soft fractal, faster mixing |

Parameters chosen so that: (a) components are distinguishable within 16 tokens (N* < 16), (b) each component produces a distinct fractal geometry in belief space.

**Sequence generation:** Each sequence of length 16 is drawn entirely from one component k ~ Uniform(3). The corpus is non-ergodic because sequences never switch components.

**Ground-truth belief states:** For every token position in every sequence, the exact belief state (π(t), η₁(t), η₂(t), η₃(t)) is computed analytically using the two-step update rule (see Section 3). These serve as regression targets in the geometry analysis.

### Architecture

Small GPT-2-style transformer via TransformerLens:
- 2 layers, d_model=64, 4 heads, d_head=16, d_mlp=256
- Context window: 16 tokens, vocabulary: {0, 1, 2}
- ~100k parameters

TransformerLens is used throughout for residual stream hook access.

### Training

- Cross-entropy next-token prediction
- AdamW, lr=1e-3 with cosine decay, 1k warmup steps
- 50k steps total, checkpoints every 5k steps
- 50k training sequences, 5k validation sequences

---

## 3. Pre-Registered Predictions

### Mathematical Derivation

**The joint belief state.** The joint hidden state space is S_joint = S₁ ⊔ S₂ ⊔ S₃ (9 states). The joint token-labeled transition matrices are **block-diagonal**:

```
T_joint^(x) = diag(T_A^(x), T_B^(x), T_C^(x))
```

The zeros off the diagonal encode non-ergodicity exactly. This block structure means that the joint belief state η_joint ∈ Δ(8) decomposes uniquely as:

```
η_joint[k, s] = π_k · η_k[s]
```

where π ∈ Δ(2) is the meta-belief and η_k ∈ Δ(2) is the within-component belief.

**The two-step update.** Upon observing token x:
- Step 1 (meta-belief): L_k(x) = η_k · T_k^(x) · 1; π'_k = π_k · L_k(x) / Z
- Step 2 (within-component, independent): η_k' = η_k · T_k^(x) / L_k(x)

Step 2 applies the standard single-Mess3 update *independently* to each component. The components couple only through the normalisation Z in Step 1.

**Conditional independence.** Conditioned on component identity k, the within-component beliefs η_j for j ≠ k evolve independently of η_k (they share the normalisation Z, but their updates are otherwise separate). This is the condition under which Shai et al. 2025 (Theorem 1) proves that factored representations are lossless: no information is lost by encoding π and η_k in separate orthogonal subspaces.

**The factored representation theorem** (informal): If the generating process has a block-diagonal structure with conditionally independent factors, then a transformer trained on next-token prediction will have an inductive bias toward representing each factor in a separate subspace, provided it can do so without capacity loss. For K=3 Mess3: the 8-dimensional belief space decomposes into 1 (2D) π subspace and 3 (2D) η_k subspaces, totalling 8 dimensions — so there is no capacity loss.

**Synchronisation horizon.** N* ≈ 1/KL_min where KL_min = min_{k≠k'} D_KL(P_k || P_k') and P_k is the stationary emission distribution of component k. We compute this analytically and verify N* < 16 for the chosen parameters.

**Dimensionality prediction.** The full belief space is 8-dimensional (dim(Δ(8)) = 8). Under H2 (factored), PCA should show exactly 8 principal components with non-negligible variance, structured as 4 pairs:
- Pair 0: π subspace (2D simplex, not fractal)
- Pairs 1-3: η_A, η_B, η_C subspaces (each a 2D Mess3 fractal)

### Intuitive Prediction

**My intuition for H2 (factored):** The block-diagonal structure of T_joint^(x) means that each component is a completely isolated information channel — it can only "talk" to itself. This isolation should make it natural for the transformer to dedicate separate weight directions to each channel. The π subspace acts like a "routing" circuit: once it collapses to one vertex (past N*), the corresponding η_k can be read out cleanly. This separation is not just possible — it's informationally free.

**Alternative geometries:**

*H1 (joint):* The transformer could represent η_joint directly as one big blob in an 8D subspace, without internal organisation. This would look like: high R² for all targets, but removing the π subspace directions would degrade η_k R² significantly. PCA would show 8 components but without the paired structure.

*H3 (superposition):* Features could be packed into fewer effective dimensions using near-orthogonal but non-orthogonal directions. This would look like: high R² but with modest cross-subspace overlap (0.1-0.3 range) and partial R² degradation in projection tests.

**My prediction:** H2 is most likely for late layers and late context positions (past N*). Early layers and early positions may show H1 or H3 structure, transitioning to H2 as training proceeds and context accumulates.

**Specific predictions (pre-registered):**
1. CEV dimensionality ≈ 8 in the final layer
2. R²(π) increases sharply around position N*
3. R²(η_k) is high (>0.9) throughout for the active component
4. Subspace overlap between π and η_k subspaces < 0.2
5. R²(η_k) drops by < 0.1 after projecting out π subspace (key H2 test)
6. π subspace emerges at earlier training steps than η_k fractal geometry

---

## 4. Methods

### Geometry Analysis Pipeline

**Residual stream extraction:** TransformerLens hooks extract the residual stream at each layer's output (`blocks.{i}.hook_resid_post`) for all N sequences.

**Regression-based subspace identification** (Shai et al. 2024): Fit a linear regression from residual stream activations to ground-truth belief targets. Extract the top singular vectors of the weight matrix — these span the *subspace* in which the belief state is encoded. R² measures how well the full subspace is recovered.

**Subspace overlap metric** (Shai et al. 2025, Appendix H): For two subspaces A (basis Q_A) and B (basis Q_B), compute the Frobenius norm of Q_A Q_B^T, normalised by √(rank_A · rank_B). This equals 0 for orthogonal subspaces and 1 for identical subspaces.

**Projection test** (key for H1 vs H2 discrimination): Project out the π subspace from residual stream activations, then re-run η_k regression. If R² drops significantly → H1. If R² is preserved → H2.

**PCA + CEV dimensionality:** Run PCA on flattened (N×L, d_model) activations. The CEV (cumulative explained variance) dimensionality is the smallest d such that the top-d components explain ≥ 95% of variance.

**Fractal visualisation:** Project activations from sequences of component k (after N*) onto the top-2 singular vectors of the η_k regression. Compare the scatter to the ground-truth MSP attractor (simulated by running the IFS for 50k steps).

**Vary-one analysis:** For each component k, fit the η_k subspace using only sequences from component k. Check cross-component subspace overlap — if components have distinct η subspaces (low cross-overlap), this confirms component-specific fractal geometry.

**Training dynamics:** Load each checkpoint (saved every 5k steps) and run regression analysis at the final layer, final position. Track R² over training to identify phase transitions.

---

## 5. Results

*[To be filled in after running the experiment. The notebook `notebooks/results.ipynb` generates all figures and computes all metrics. Key figures to include: `msp_fractals.png`, `training_loss.png`, `pca_explained_variance.png`, `r2_vs_position_all_layers.png`, `subspace_overlap.png`, `fractal_comparison.png`, `training_dynamics.png`.]*

### 5.1 Training Loss

*[Include training_loss.png. Discuss: does loss decrease? Is per-position loss lower for later positions (more context)? Is there a loss "bump" at induction head formation?]*

### 5.2 PCA Dimensionality

*[Include pca_explained_variance.png. Does the elbow occur at 8 components? Is there a paired structure?]*

### 5.3 R² by Layer and Context Position

*[Include r2_vs_position_all_layers.png. Key question: does π R² increase sharply at N*? Is η_k R² high throughout?]*

### 5.4 Subspace Orthogonality

*[Include subspace_overlap.png and projection test table. Key test: is R² drop after removing π < 0.1?]*

### 5.5 Fractal Geometry

*[Include fractal_comparison.png. Does the residual stream projection match the MSP fractal?]*

### 5.6 Training Dynamics

*[Include training_dynamics.png. Does π subspace emerge earlier than η_k?]*

---

## 6. Additional Analysis: Vary-One

**Why this analysis?** The vary-one analysis tests whether the learned η_k subspaces are component-specific, not just a single undifferentiated "within-component" subspace. For H2, each η_k should live in its own 2D subspace, and sequences from component k should use that subspace exclusively. If we fit the η_k subspace using only sequences from component k, and find that this subspace has low overlap with the η_j subspace fit from component j sequences (j ≠ k), this would confirm that the transformer has learned three distinct fractal geometries — one per component.

**Method:** Separate sequences by true component identity (using component_ids). For each k, fit η_k regression using only the activations from component k sequences. Compute pairwise subspace overlap between the three resulting η subspaces.

**Prediction under H2:** Cross-component subspace overlap should be low (< 0.2). Under H1, the overlap would be higher because the model uses one shared subspace.

*[Results to be filled after running the experiment.]*

---

## 7. Discussion

### Which Hypothesis Best Fits?

*[To be filled after results.]*

Under H2 (predicted), we would observe:
- High R² for both π and η_k (> 0.9) in the final layer past N*
- Low subspace overlap between π and η_k directions (< 0.2)
- Near-zero R² drop in projection tests
- CEV dimensionality ≈ 8, with paired structure
- Distinct fractal geometry for each component in its subspace
- π subspace emerging before η_k fractal structure in training

### Implications for Language Models

If H2 holds, this suggests that:

1. **Non-ergodic pretraining naturally induces factored representations.** The transformer does not need to be explicitly supervised to separate "what domain am I in?" from "what is happening within this domain?" The block structure of the generative process is enough.

2. **Two-level belief structure maps onto two-phase training dynamics.** The π subspace (meta-belief, fast) should emerge before η_k (within-component, slower) — consistent with the "concept formation then architectural reorganisation" pattern observed in other work.

3. **In-context learning has geometric structure.** ICL in real language models may involve a π-like circuit that routes processing to domain-specific η-like circuits, which is exactly the factored structure found here.

4. **Superposition and factoring coexist.** While H2 predicts orthogonal subspaces, the limited model size (d=64) means the model may use some superposition within each subspace to store more than 2 effective dimensions per factor. Investigating whether within-subspace superposition exists is a natural next step.

### Limitations

- Small model (d=64) and short context (L=16) may limit generality
- Symmetric component structure (same alphabet) may make factoring easier
- Ground-truth belief computation uses exact analytically-derived updates — real corpora do not provide this

---

## References

1. Shai, A., et al. (2024). *Transformers represent belief state geometry in their residual stream.* NeurIPS 2024. arXiv:2405.15943.

2. Riechers, P., et al. (2025). *Next-token pretraining implies in-context learning.* arXiv:2505.18373.

3. Shai, A., et al. (2025). *Transformers learn factored representations.* ICML 2025. arXiv:2602.02385.
