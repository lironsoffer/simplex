# Plan: Coupling Sweep — Factored vs. Joint Representations Under ε-Coupling

## Motivation

The factored representation result (Shai et al. ICML 2025) guarantees lossless factorization *if and only if* the factors are conditionally independent. In our setup, the K=3 Mess3 components are exactly conditionally independent (block-diagonal joint transition matrix), so the factored geometry is theoretically optimal.

The natural follow-up: **what happens as we break conditional independence?**

The information-theoretic quantity that measures the loss from factoring is:

```
I(η_k ; η_k' | π)
```

At ε=0 (block-diagonal), this is exactly zero — factoring is lossless. As off-diagonal coupling ε grows, this increases, meaning the factored representation throws away real cross-component correlations. The prediction: the transformer's geometry should transition from factored orthogonal subspaces toward a joint entangled representation as ε increases.

This is a clean phase-transition experiment with a principled information-theoretic axis.

---

## Two Implementation Paths

### Path A: Full Sweep (retrain for each ε)

Train separate transformers on datasets generated with ε ∈ {0, 0.02, 0.05, 0.10}. For each trained model, measure subspace overlap and cross-component regression R².

**Advantages:** Clean, controls for everything, strongest result.
**Disadvantages:** 4× compute budget, requires re-deriving belief update math for ε > 0.

### Path B: Test-Time Coupling Shortcut (no retraining)

Take the **already-trained** model (ε=0) and evaluate it on sequences generated with small coupling ε injected at test time only. Measure how geometry and prediction accuracy degrade as ε increases.

**Advantages:** No retraining, fast to run, reuses all existing analysis tools.
**Disadvantages:** Weaker claim — tests robustness of a fixed representation, not whether a model *trained* under coupling would learn differently.

---

## Path B: Detailed Design (Alternative Shortcut)

### Core Idea

The trained model learned a factored representation under exact conditional independence. We ask: **how robust is this representation when the test distribution violates the training assumption?**

Concretely, we generate test sequences from a *coupled* joint HMM (ε > 0) and pass them through the frozen model. We then:
1. Compare model predictions to the correct ground-truth next-token distribution under the coupled process.
2. Extract residual stream activations and run the same regression/orthogonality analysis as before.
3. Track how geometry and loss degrade as ε increases.

### Coupled HMM Construction

The current joint transition matrix is block-diagonal:

```
T_joint(ε=0) = diag(T_A, T_B, T_C)
```

Introduce coupling by adding a small off-diagonal mass:

```
T_joint(ε) = (1 - ε) · T_joint(0) + ε · T_leak
```

where `T_leak` is a "leakage" matrix that allows cross-component transitions (e.g., uniform over all states, or directed toward the nearest component's stationary distribution).

At ε=0: exact block-diagonal (training distribution).
At ε=1: fully coupled ergodic process (far OOD).

### Belief Update Under Coupling

For ε > 0, the clean two-step factored update breaks. The correct belief state is now a full joint distribution over (k, state_k). To compute ground-truth belief states for regression targets, track the full joint belief:

```
b_t = distribution over (component k, within-component state s_k)
b_{t+1}(k', s') = Σ_{k,s} b_t(k,s) · T_joint(ε)[(k,s) → (k',s')] · emission(x_{t+1} | k', s') / Z
```

This is a 3×3 = 9-dimensional belief state (for 3 components × 3 states each). Extract the marginals:
- π_k = Σ_s b(k, s) — meta-belief
- η_k(s) = b(k, s) / π_k — within-component beliefs

### Metrics to Track vs. ε

| Metric | What it measures |
|--------|-----------------|
| Per-token cross-entropy loss (model vs. true ε-process) | Prediction accuracy degradation |
| R²(residual → π) | Does meta-belief encoding survive? |
| R²(residual → η_k) | Does within-component encoding survive? |
| Subspace overlap(π subspace, η_k subspace) | Does orthogonality break down? |
| I(η_k ; η_k' \| π) empirical estimate | Ground-truth coupling axis |

### I(η_k ; η_k' | π) Estimation

Use a simple binned estimator over the belief state trajectories:
1. Generate N=10k trajectories at each ε under the coupled process.
2. Bin π into K=3 buckets (which component dominates).
3. Within each bucket, compute the empirical covariance of (η_k, η_k').
4. Use a k-NN mutual information estimator (e.g., from `scipy` or `sklearn`).

This gives the ground-truth x-axis for plotting geometry metrics against.

### Expected Results

- **ε = 0:** factored geometry, orthogonal subspaces, high R² for both π and η_k.
- **Small ε (0.01–0.05):** geometry largely preserved, loss increases slightly. The factored representation is robust to small perturbations.
- **Large ε (0.10+):** geometry degrades. If the factored subspaces are truly insufficient, R²(η_k) should drop faster than R²(π), since cross-component η correlations are what the factored representation discards.

The interesting question: is there a **sharp transition** (phase-transition-like), or a gradual degradation?

### What the Shortcut *Cannot* Claim

This path tests robustness of a fixed ε=0 representation. It cannot answer: *would a model trained at ε=0.05 learn a different geometry?* That requires Path A. The shortcut is best framed as: "we probe the limits of the learned factored representation."

---

## Implementation Checklist

### Path B

- [ ] Implement `CoupledMess3HMM(epsilon)` in `src/data/mess3.py`
- [ ] Implement joint belief update in `src/data/belief_update.py`
- [ ] Generate test sequences at ε ∈ {0.0, 0.01, 0.02, 0.05, 0.10, 0.20} — no retraining
- [ ] Add `run_coupling_sweep()` to `src/analysis/` — loops over ε, extracts activations, runs regression + orthogonality checks
- [ ] Implement `estimate_conditional_mi()` — empirical I(η_k ; η_k' | π)
- [ ] Plot: subspace overlap, R²(π), R²(η_k) vs. ε on shared axes
- [ ] Plot: I(η_k ; η_k' | π) vs. ε as the ground-truth coupling axis

### Path A (if time permits)

- [ ] All of Path B data generation
- [ ] Retrain transformer at each ε (4 runs)
- [ ] Run full geometry analysis at each checkpoint

---

## Open Questions

1. Does the factored representation degrade gracefully or sharply with ε?
2. Does the π subspace survive longer than the η_k subspace as coupling increases? (Prediction: yes — π is a coarser summary and should be more robust.)
3. Is there a ε* where the model's representation transitions from "approximately factored" to "clearly joint"?
4. Does this threshold correspond to a detectable change in I(η_k ; η_k' | π)?
