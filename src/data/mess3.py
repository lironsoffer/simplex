"""
Mess3 HMM: 3-state, 3-token edge-emitting Hidden Markov Model.

Parameters:
    alpha: emission strength in (0.5, 1) — how strongly each state prefers its token
    x: mixing rate in (0, 1/3) — probability of transitioning to each other state

Hidden states: S = {0, 1, 2}
Token alphabet: X = {0, 1, 2}

Token-labeled transition matrix T^(a)[i, j]:
    P(token=a, next_state=j | current_state=i)
    = P(next_state=j | state=i) * P(token=a | state=i)

where:
    P(next_state=j | state=i) = (1 - 2x) if i == j, else x
    P(token=a | state=i)      = alpha if a == i, else (1 - alpha) / 2
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Mess3Params:
    alpha: float
    x: float
    name: str = ""

    def __post_init__(self):
        if not (0.5 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0.5, 1), got {self.alpha}")
        if not (0.0 < self.x < 1.0 / 3.0):
            raise ValueError(f"x must be in (0, 1/3), got {self.x}")


class Mess3HMM:
    """
    Mess3 Hidden Markov Model with token-labeled transition matrices.

    The model has 3 hidden states and 3 tokens. Token a is most likely
    emitted by state a (with probability alpha), and transition structure
    keeps the HMM near its current state (with probability 1-2x).
    """

    N_STATES = 3
    N_TOKENS = 3

    def __init__(self, params: Mess3Params):
        self.params = params
        self.alpha = params.alpha
        self.x = params.x
        self._transition_matrices = self._build_transition_matrices()
        self._stationary = self._compute_stationary()

    def _build_transition_matrices(self) -> np.ndarray:
        """
        Build token-labeled transition matrices T^(a) for a in {0, 1, 2}.

        Returns:
            T: shape (3, 3, 3) where T[a, i, j] = P(token=a, next_state=j | state=i)
        """
        alpha = self.alpha
        x = self.x
        n = self.N_STATES

        # P(next_state=j | state=i)
        transition = np.full((n, n), x)
        np.fill_diagonal(transition, 1.0 - 2 * x)

        # P(token=a | state=i)
        emission = np.full((n, n), (1.0 - alpha) / 2.0)
        np.fill_diagonal(emission, alpha)

        # T^(a)[i, j] = P(transition i->j) * P(emit a | state i)
        # Shape: (n_tokens, n_states, n_states)
        T = np.zeros((n, n, n))
        for a in range(n):
            for i in range(n):
                T[a, i, :] = transition[i, :] * emission[i, a]

        # Verify: sum over tokens and next states = 1 for each current state
        assert np.allclose(T.sum(axis=(0, 2)), 1.0), "Transition matrices must sum to 1"
        return T

    def _compute_stationary(self) -> np.ndarray:
        """
        Compute stationary distribution of the full transition matrix (summed over tokens).

        Returns:
            stationary: shape (3,), sums to 1
        """
        # Full transition matrix P[i, j] = sum_a T^(a)[i, j]
        P = self._transition_matrices.sum(axis=0)

        # Solve stationary: pi @ P = pi, sum(pi) = 1
        # Equivalent to left eigenvector for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = eigenvectors[:, idx].real
        stationary = stationary / stationary.sum()

        assert np.allclose(stationary @ P, stationary, atol=1e-10), "Not stationary"
        assert np.allclose(stationary.sum(), 1.0), "Not normalized"
        assert np.all(stationary >= 0), "Negative probabilities"

        return stationary

    @property
    def transition_matrices(self) -> np.ndarray:
        """Token-labeled transition matrices, shape (3, 3, 3): T[token, from_state, to_state]."""
        return self._transition_matrices

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Stationary distribution over hidden states, shape (3,)."""
        return self._stationary.copy()

    def update_belief(self, eta: np.ndarray, token: int) -> np.ndarray:
        """
        Bayesian update of belief state after observing token.

        eta'[j] = sum_i eta[i] * T^(token)[i, j] / P(token | eta)

        Args:
            eta: current belief state, shape (3,), sums to 1
            token: observed token in {0, 1, 2}

        Returns:
            eta_new: updated belief state, shape (3,), sums to 1
        """
        T_a = self._transition_matrices[token]
        unnorm = eta @ T_a
        Z = unnorm.sum()
        return unnorm / Z

    def generate_sequence(
        self,
        length: int,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a sequence of tokens and corresponding hidden states.

        Args:
            length: sequence length
            rng: random number generator (default: new generator)

        Returns:
            tokens: shape (length,), values in {0, 1, 2}
            states: shape (length,), values in {0, 1, 2}
        """
        if rng is None:
            rng = np.random.default_rng()

        # Start in state drawn from stationary distribution
        state = rng.choice(self.N_STATES, p=self._stationary)
        tokens = np.zeros(length, dtype=np.int64)
        states = np.zeros(length, dtype=np.int64)

        for t in range(length):
            # Emit token based on current state
            emission_probs = np.array([
                self._transition_matrices[a, state, :].sum()
                for a in range(self.N_TOKENS)
            ])
            # emission_probs[a] = sum_j T^(a)[state, j] = P(token=a | state)
            token = rng.choice(self.N_TOKENS, p=emission_probs)
            tokens[t] = token
            states[t] = state

            # Transition to next state based on token and current state
            next_state_probs = self._transition_matrices[token, state, :]
            next_state_probs = next_state_probs / next_state_probs.sum()
            state = rng.choice(self.N_STATES, p=next_state_probs)

        return tokens, states

    def log_likelihood_per_token(
        self,
        tokens: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Compute average per-token log-likelihood of a token sequence under this HMM.

        Uses the forward algorithm (belief propagation) for exact computation.

        Args:
            tokens: (T,) int64 token sequence
            rng: unused, kept for API compatibility

        Returns:
            Average log P(token_t | token_1, ..., token_{t-1}) over t=1..T
        """
        eta = self._stationary.copy()
        total_log_prob = 0.0

        for token in tokens:
            T_a = self._transition_matrices[token]
            unnorm = eta @ T_a
            Z = unnorm.sum()
            if Z < 1e-15:
                break
            total_log_prob += np.log(Z)
            eta = unnorm / Z

        return total_log_prob / len(tokens)

    def kl_divergence_rate_from(
        self,
        other: "Mess3HMM",
        n_sequences: int = 500,
        seq_length: int = 200,
        seed: int = 0,
    ) -> float:
        """
        Estimate per-token KL divergence rate D_KL(self || other).

        D_KL(P_self || P_other) = lim_{T→∞} (1/T) * E_{x~P_self}[log P_self(x) - log P_other(x)]

        Estimated empirically by generating sequences from self and computing
        average per-token log-likelihood ratio.

        Args:
            other: another Mess3HMM
            n_sequences: number of sequences to sample
            seq_length: length of each sequence (longer = lower variance)
            seed: random seed

        Returns:
            Estimated per-token KL divergence rate (≥ 0)
        """
        rng = np.random.default_rng(seed)
        kl_rates = []

        for _ in range(n_sequences):
            tokens, _ = self.generate_sequence(seq_length, rng=rng)
            ll_self = self.log_likelihood_per_token(tokens)
            ll_other = other.log_likelihood_per_token(tokens)
            kl_rates.append(ll_self - ll_other)

        kl = float(np.mean(kl_rates))
        return max(0.0, kl)  # KL must be non-negative


# Default component parameters
# Chosen so that N* ≈ 10 < 16 (context window), ensuring synchronisation within a sequence.
# Component C (α=0.55) produces a near-degenerate attractor (~9 visually distinct clusters),
# which is mathematically correct for these parameters.
COMPONENT_PARAMS = [
    Mess3Params(alpha=0.95, x=0.02, name="A"),  # sharp fractal, slow mixing
    Mess3Params(alpha=0.80, x=0.08, name="B"),  # medium fractal, moderate mixing
    Mess3Params(alpha=0.55, x=0.25, name="C"),  # sparse attractor, fastest mixing
]


def build_default_components() -> list[Mess3HMM]:
    """Build the three default Mess3 components."""
    return [Mess3HMM(p) for p in COMPONENT_PARAMS]


def compute_synchronisation_horizon(
    components: list[Mess3HMM],
    n_sequences: int = 500,
    seq_length: int = 200,
    seed: int = 0,
) -> float:
    """
    Compute synchronisation horizon N* = 1 / min_{k!=k'} D_KL(P_k || P_k').

    N* is the expected number of tokens needed for the meta-belief π to
    concentrate on the true component. Should be less than the context window
    length (16) for the components to be distinguishable within a sequence.

    Uses empirical per-token KL divergence rates (not 1-gram marginals, which
    are identical for all Mess3 components due to uniform stationary distribution).

    Args:
        components: list of Mess3HMM instances
        n_sequences: sequences per pair for KL estimation
        seq_length: sequence length per KL estimation
        seed: random seed

    Returns:
        N*: synchronisation horizon
    """
    K = len(components)
    min_kl = float("inf")

    for i in range(K):
        for j in range(K):
            if i != j:
                kl = components[i].kl_divergence_rate_from(
                    components[j], n_sequences=n_sequences,
                    seq_length=seq_length, seed=seed + i * K + j,
                )
                if kl > 1e-10:
                    min_kl = min(min_kl, kl)

    if min_kl == float("inf"):
        return float("inf")
    return 1.0 / min_kl
