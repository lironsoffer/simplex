# Theoretical Background: Belief State Geometry in Non-Ergodic Transformers

## 1. The Single Mess3 Process

### Definition

A Mess3 process is a 3-state, 3-token edge-emitting Hidden Markov Model (HMM) with two parameters:
- **α ∈ (0.5, 1):** emission strength — how strongly each state prefers its own token
- **x ∈ (0, 1/3):** mixing rate — how likely the hidden state is to transition

Hidden states: S = {0, 1, 2}. Tokens: X = {0, 1, 2}.

The token-labeled transition matrices T^(k) have entries:

```
T^(k)[i, j] = P(token=k, next_state=j | current_state=i)
             = P(next_state=j | state=i) · P(token=k | state=i)

where:
  P(next_state=j | state=i) = (1-2x)  if i=j
                               x        if i≠j

  P(token=k | state=i)      = α        if k=i
                               (1-α)/2  if k≠i
```

### Belief State

The belief state η(t) is a probability distribution over hidden states given observed context:

```
η(t) = (η₀, η₁, η₂)    where ηᵢ = P(state=i | x₁, ..., xₜ)
```

η(t) is a point in the 2-simplex Δ(2) — a triangle whose vertices represent certainty about each state.

### Bayesian Update

Upon observing token x at time t+1:

```
η(t+1) = η(t) · T^(x) / (η(t) · T^(x) · 1)
```

This is a contraction map fₓ : Δ(2) → Δ(2). It takes any belief state and moves it toward the region of the simplex consistent with having just seen token x.

### The IFS and the Fractal

The three maps {f₀, f₁, f₂} form an Iterated Function System (IFS). Each map is a contraction that pulls beliefs toward a different region of the simplex:
- f₀ pulls toward vertex 0 (token 0 is evidence for state 0)
- f₁ pulls toward vertex 1
- f₂ pulls toward vertex 2

The **Mixed State Presentation (MSP)** is the unique attractor of this IFS — the set F satisfying:

```
F = f₀(F) ∪ f₁(F) ∪ f₂(F)
```

F is a fractal: a self-similar subset of the simplex. Any belief trajectory starting from the stationary distribution converges to F and stays there. A sequence of T tokens traces a path of T points on F, landing deeper in the fractal structure as T increases.

**Effect of parameters on fractal shape:**
- High α, low x → maps are strong contractions, fractal reaches toward vertices, fine detail, sparse
- Low α, high x → maps are weak contractions, fractal collapses toward centre, coarse, dense

### Key Result (Shai et al. NeurIPS 2024)

A transformer trained on Mess3 sequences via next-token prediction linearly represents η(t) in its residual stream. The set of residual stream activations, projected onto a 2D subspace, forms the MSP fractal. This holds across training and layers.

---

## 2. The Non-Ergodic Extension: K Mess3 Processes

### Why Non-Ergodic?

A single Mess3 is ergodic — given infinite time it visits all regions of its fractal. A corpus of K Mess3 processes, where each sequence is generated entirely by one process, is **non-ergodic**: the generating process never transitions between components. A sequence from component A stays in component A forever.

This models real language corpora: each document belongs to one genre/domain/author (component), never switching mid-document.

### The Joint HMM

The joint hidden state space is the disjoint union:

```
S_joint = S₁ ⊔ S₂ ⊔ ... ⊔ S_K     (total: 3K states)
```

State (k, s) means "in component k, hidden state s."

The joint token-labeled transition matrices are **block-diagonal**:

```
T_joint^(x) = diag(T₁^(x), T₂^(x), ..., T_K^(x))
```

The zeros off the diagonal are exact — there is zero probability of transitioning between components. This block-diagonal structure is the mathematical expression of non-ergodicity.

### The Joint Belief State

The joint belief η_joint ∈ Δ(3K-1) is a distribution over all 3K states:

```
η_joint[k, s] = P(component=k, hidden_state=s | context)
```

Update rule (same form as single component):

```
η_joint' = η_joint · T_joint^(x) / (η_joint · T_joint^(x) · 1)
```

Block-diagonal structure means the k-th block only interacts with T_k^(x).

---

## 3. The Two-Level Decomposition

### Definition

The joint belief decomposes exactly into:

**Meta-belief π ∈ Δ(K-1):**
```
πₖ = Σ_s η_joint[k, s]     (marginal probability of component k)
```

**Within-component belief η_k ∈ Δ(2):**
```
η_k[s] = η_joint[k, s] / πₖ     (conditional distribution over states within k)
```

**Reconstruction (always exact):**
```
η_joint[k, s] = πₖ · η_k[s]
```

This is a bijection — (π, η₁, ..., η_K) contains exactly the same information as η_joint, just written differently.

### The Two-Step Update Rule

Upon observing token x:

**Step 1 — Meta-belief update (Bayesian model selection):**
```
L_k(x)  = η_k · T_k^(x) · 1          [likelihood of token x under component k]
Z        = Σ_k πₖ · L_k(x)            [total probability of token x]
π'ₖ      = πₖ · L_k(x) / Z            [updated meta-belief — Bayes' rule]
```

**Step 2 — Within-component update (independent per component):**
```
η_k'     = η_k · T_k^(x) / L_k(x)    [standard single-component Mess3 update]
```

**Key structural facts:**
1. Step 2 is identical to the single-component Mess3 update, applied independently to each k
2. The two steps couple only through L_k(x) — the shared likelihood
3. All K within-component updates run simultaneously, even for components with πₖ ≈ 0
4. Once πₖ = 1 for some k, π is frozen and only η_k continues to evolve

### Conditional Independence

The block-diagonal structure of T_joint^(x) implies: conditioned on component identity k, the within-component beliefs η_k for different components are **statistically independent**. They never interact except through the shared normalisation in Step 1.

This is the condition under which the factored representations paper (2602.02385) proves factored representations are lossless.

### Dimensionality

```
η_joint  ∈ Δ(3K-1):            dimension = 3K-1
π        ∈ Δ(K-1):             dimension = K-1
η_k      ∈ Δ(2) each:          dimension = 2  (per component)
(π, η_k) ∈ Δ(K-1) × Δ(2)^K:   dimension = (K-1) + 2K = 3K-1
```

For K=3: all representations are 8-dimensional. No dimension reduction from factoring — but the structure is interpretably organised.

---

## 4. Geometry of Each Object

### η_k: A Mess3 Fractal

Each within-component belief η_k traces the MSP fractal of component k. The fractal's shape is determined by (αₖ, xₖ). Different components have different fractal geometries — this is what makes them distinguishable.

Each η_k individually traces a 2D fractal, living in a 2D subspace of the full representation space.

### π: A Simplex, Not a Fractal

π lives in Δ(K-1) — a (K-1)-simplex. For K=3 this is a triangle. Unlike η_k, π does not have fractal structure. Its dynamics are:

- At t=0: uniform (1/K, ..., 1/K) — no information about which component
- As tokens arrive: π moves toward one vertex at rate determined by KL divergence between components
- Past N*: π ≈ eₖ (one-hot on the true component) and stays there

The synchronisation horizon N* controls how quickly π collapses:
```
N* ≈ 1 / min_{k≠k'} D_KL(P_k || P_k')
```

where P_k is the stationary emission distribution of component k. Choose (αₖ, xₖ) so that N* < context window length.

---

## 5. Competing Representational Hypotheses

### H1 — Joint Representation

The residual stream represents η_joint as one undifferentiated (3K-1)-dimensional blob. No internal organisation aligns with the (π, η_k) decomposition. Removing directions associated with π also removes information needed for η_k.

### H2 — Factored Representation (predicted)

The residual stream organises its 3K-1 dimensions into K+1 orthogonal subspaces:

```
Subspace 0:  (K-1)-dimensional, encodes π
Subspace k:  2-dimensional, encodes η_k, contains component k's fractal   (for k=1,...,K)
```

Predicted by: block-diagonal joint HMM → conditional independence → factored representation theorem (Shai et al. 2602.02385).

### H3 — Superposition

Multiple features packed into fewer dimensions via non-orthogonal directions. π and η_k are all recoverable approximately but with interference. Directions found by regressing to π overlap with directions found by regressing to η_k.

---

## 6. Testable Predictions (Under H2)

| Measurement | Prediction |
|---|---|
| PCA dimensionality | 8 non-negligible components, grouped as 4 pairs |
| R²(residual → π) | ≈ 1, especially past N* |
| R²(residual → η_k) | ≈ 1, for the active component |
| Cosine sim(π subspace, η_k subspace) | ≈ 0 for all k |
| R²(η_k) after projecting out π subspace | Still ≈ 1 (key test vs H1) |
| Fractal in η_k projection | Matches ground-truth MSP for component k's (α,x) |
| R²(π, t) vs context position | Increases sharply around N* |
| Layer of π subspace emergence | Earlier layers than η_k fractal |

---

## 7. Connection to Related Papers

### Shai et al. NeurIPS 2024 — "Transformers represent belief state geometry"
[arXiv:2405.15943](https://arxiv.org/abs/2405.15943) | [Code: adamimos/epsilon-transformers](https://github.com/adamimos/epsilon-transformers/tree/main)

Single ergodic Mess3. Establishes: (1) transformer linearly encodes η(t), (2) the fractal geometry is present in residual stream, (3) regression method for measuring this. **Your experiment extends this to K components.**

### Riechers et al. 2025 — "Next-token pretraining implies in-context learning"
[arXiv:2505.18373](https://arxiv.org/abs/2505.18373)

Proves ICL emerges necessarily from non-ergodic pretraining. Key results: (1) non-ergodic sources produce power-law loss decay (not exponential), (2) synchronisation horizon N* = 1/KL, (3) induction heads implement component disambiguation. **Your experiment uses the same training setup but measures geometry rather than loss dynamics.**

### Shai et al. ICML 2025 — "Transformers learn factored representations"
[arXiv:2602.02385](https://arxiv.org/abs/2602.02385) | [Code: Astera-org/factored-reps](https://github.com/Astera-org/factored-reps/tree/master)

K processes running **in parallel** (joint tokens from 3 Mess3 + 2 Bloch Walk). Finds orthogonal subspaces in residual stream. Provides: (1) competing hypotheses framework, (2) vary-one analysis method, (3) subspace overlap metric, (4) CEV dimensionality measure. **Your experiment uses the same geometric question but with temporal mixture instead of parallel factors.**

### Community reimplementation
[sanowl/BeliefStateTransformer](https://github.com/sanowl/BeliefStateTransformer/tree/main) — extends epsilon-transformers with layer-wise analysis and training dynamics tracking. Useful reference if the official repo is hard to navigate.

### The Gap This Project Fills

```
                    | Ergodic (single)  | Temporal mixture (K sources) | Parallel factors (K sources)
--------------------|-------------------|------------------------------|-----------------------------
Loss dynamics       | —                 | Riechers 2025 ✓              | —
Residual geometry   | Shai 2024 ✓       | THIS PROJECT                 | Shai 2025 ✓
```

---

## 8. Connection to ICLR Paper (Liron et al. 2026)

The two-phase training dynamic found in "From Tokens to Thoughts":
- **Phase 1:** Rapid initial concept formation — models form representations of semantic content quickly
- **Phase 2:** Architectural reorganisation — semantic processing migrates from deep to mid-network layers

Maps onto the two-level belief structure:
- **Phase 1 ↔ Meta-belief π:** rapid identification of which component/concept is active (induction head formation, component disambiguation)
- **Phase 2 ↔ Within-component η_k:** architectural reorganisation into separate subspaces as the model learns to efficiently track within-component belief state

The Simplex prediction of factored orthogonal subspaces provides a geometric account of why the phases are distinct and why processing migrates between layers.
