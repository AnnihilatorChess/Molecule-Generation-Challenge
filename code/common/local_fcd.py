"""Local FCD harness — compute and reuse (mean, cov) of ChemNet activations on a
fixed held-out slice of training SMILES, then evaluate generated SMILES against it.

This lets us pick models without burning submissions.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

# CPU-only inside ChemNet — matches the official evaluator
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import fcd  # noqa: E402


def _load_model():
    return fcd.load_ref_model()


def compute_stats(smiles: list[str], model=None) -> tuple[np.ndarray, np.ndarray]:
    """ChemNet activations → (mean, cov). Matches `evaluation.utils.getstats`."""
    if model is None:
        model = _load_model()
    preds = fcd.get_predictions(model, smiles)
    mu = preds.mean(0)
    cov = np.cov(preds.T)
    return mu, cov


def save_ref(mu: np.ndarray, cov: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((mu, cov), f)


def load_ref(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)


def local_fcd(
    generated: list[str], ref_path: Path, model=None
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """FCD of `generated` against the saved local reference. Returns (fcd, (mu_gen, cov_gen))."""
    mu_ref, cov_ref = load_ref(ref_path)
    mu_gen, cov_gen = compute_stats(generated, model=model)
    val = fcd.calculate_frechet_distance(
        mu1=mu_gen, mu2=mu_ref, sigma1=cov_gen, sigma2=cov_ref
    )
    return float(val), (mu_gen, cov_gen)
