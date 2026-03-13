# Belief State Geometry in Non-Ergodic Transformers

Mechanistic interpretability research on how transformers represent belief states when trained on non-ergodic data. We train a small transformer on a mixture of Mess3 HMM processes, where each sequence comes from one process, and analyze the geometry of its residual stream activations. See [`writeup/report.pdf`](writeup/report.pdf) for the full writeup.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python experiments/run_experiment.py --phase data
python experiments/run_experiment.py --phase train --device mps   # or cuda/cpu
python experiments/run_experiment.py --phase analyze --device mps
```

## Structure

```
src/data/         # Mess3 HMM, belief updates, dataset generation
src/model/        # TransformerLens HookedTransformer
src/training/     # Training loop and checkpointing
src/analysis/     # Regression, PCA, orthogonality, context/training dynamics
experiments/      # End-to-end runner
writeup/          # LaTeX report
```

## References

- Shai et al. 2024 — [Transformers represent belief state geometry in their residual stream](https://arxiv.org/abs/2405.15943)
- Riechers et al. 2025 — [Next-token pretraining implies in-context learning](https://arxiv.org/abs/2505.18373)
- Shai et al. 2025 — [Transformers learn factored representations](https://arxiv.org/abs/2602.02385)
