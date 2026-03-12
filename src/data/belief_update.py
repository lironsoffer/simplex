"""
Two-step belief update rule for the joint K-component Mess3 process.

The joint belief decomposes into:
    - pi: meta-belief in Delta(K-1), pi_k = P(component=k | context)
    - eta_k: within-component belief in Delta(2), eta_k[s] = P(state=s | component=k, context)

Update upon observing token x:
    Step 1 (meta-belief):
        L_k(x) = eta_k @ T_k^(x) @ 1   [likelihood of x under component k]
        Z = sum_k pi_k * L_k(x)
        pi'_k = pi_k * L_k(x) / Z

    Step 2 (within-component, independent per k):
        eta_k' = eta_k @ T_k^(x) / L_k(x)
"""

import numpy as np
from typing import NamedTuple

from src.data.mess3 import Mess3HMM


class JointBelief(NamedTuple):
    """
    Joint belief state for K-component Mess3 mixture.

    Attributes:
        pi: meta-belief, shape (K,), sums to 1
        eta: within-component beliefs, shape (K, 3), each row sums to 1
    """
    pi: np.ndarray   # (K,)
    eta: np.ndarray  # (K, 3)


def initial_belief(components: list[Mess3HMM]) -> JointBelief:
    """
    Construct initial joint belief at t=0 (no tokens observed).

    - Uniform meta-belief: pi_k = 1/K for all k
    - Within-component beliefs: stationary distribution for each component

    Args:
        components: list of K Mess3HMM instances

    Returns:
        Initial JointBelief
    """
    K = len(components)
    pi = np.full(K, 1.0 / K)
    eta = np.stack([comp.stationary_distribution for comp in components])  # (K, 3)
    return JointBelief(pi=pi, eta=eta)


def update_belief(
    belief: JointBelief,
    token: int,
    components: list[Mess3HMM],
) -> JointBelief:
    """
    Apply the two-step belief update upon observing a token.

    Args:
        belief: current joint belief (pi, eta)
        token: observed token in {0, 1, 2}
        components: list of K Mess3HMM instances

    Returns:
        Updated JointBelief
    """
    K = len(components)
    ones = np.ones(3)

    # Step 1: compute likelihoods L_k(token) for each component
    L = np.zeros(K)
    for k, comp in enumerate(components):
        T_a = comp.transition_matrices[token]  # (3, 3)
        L[k] = belief.eta[k] @ T_a @ ones

    # Normalise meta-belief
    Z = belief.pi @ L
    if Z < 1e-12:
        # Numerical underflow: return current belief unchanged
        return belief

    pi_new = belief.pi * L / Z

    # Step 2: update within-component beliefs independently
    eta_new = np.zeros_like(belief.eta)
    for k, comp in enumerate(components):
        T_a = comp.transition_matrices[token]  # (3, 3)
        if L[k] < 1e-12:
            # Component k has zero likelihood — belief stays unchanged
            eta_new[k] = belief.eta[k].copy()
        else:
            eta_new[k] = belief.eta[k] @ T_a / L[k]

    return JointBelief(pi=pi_new, eta=eta_new)


def compute_belief_trajectory(
    tokens: np.ndarray,
    components: list[Mess3HMM],
) -> list[JointBelief]:
    """
    Compute the full belief state trajectory for a token sequence.

    Args:
        tokens: shape (T,), token indices in {0, 1, 2}
        components: list of K Mess3HMM instances

    Returns:
        beliefs: list of T+1 JointBelief states
                 beliefs[0] = initial belief (before seeing any tokens)
                 beliefs[t+1] = belief after observing tokens[0..t]
    """
    belief = initial_belief(components)
    beliefs = [belief]

    for token in tokens:
        belief = update_belief(belief, int(token), components)
        beliefs.append(belief)

    return beliefs


def beliefs_to_arrays(
    beliefs: list[JointBelief],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert list of JointBelief to numpy arrays for regression targets.

    Args:
        beliefs: list of T+1 JointBelief states

    Returns:
        pi_traj: shape (T+1, K)
        eta_traj: shape (T+1, K, 3)
    """
    pi_traj = np.stack([b.pi for b in beliefs])     # (T+1, K)
    eta_traj = np.stack([b.eta for b in beliefs])   # (T+1, K, 3)
    return pi_traj, eta_traj
