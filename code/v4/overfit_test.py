"""v4 sanity check: does swap_full overfit the local reference?

Builds a SECOND local FCD reference from a fresh 10k of training SMILES
(disjoint from the original held-out slice used as ref1). Evaluates each
v4 submission file against BOTH refs. If a method scores much better on
ref1 than ref2, that method is fitting ref1's finite-sample noise rather
than the true distribution.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import fcd  # noqa: E402

from common.local_fcd import compute_stats, load_ref  # noqa: E402
from common.paths import ARTIFACTS_DIR  # noqa: E402


def read_lines(p: Path) -> list[str]:
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def fcd_val(sub_smiles, mu_r, cov_r):
    chemnet = fcd.load_ref_model()
    mu, cov = compute_stats(sub_smiles, model=chemnet)
    return float(fcd.calculate_frechet_distance(mu1=mu, mu2=mu_r, sigma1=cov, sigma2=cov_r))


def main() -> None:
    ref1_path = ARTIFACTS_DIR / "local_fcd_ref.pkl"
    mu1, cov1 = load_ref(ref1_path)
    print(f"loaded ref1 from {ref1_path}")

    # Build ref2 from a fresh 10k of train_smiles.txt
    train_file = ARTIFACTS_DIR / "v0" / "train_smiles.txt"
    train_smiles = read_lines(train_file)
    rng = np.random.default_rng(9999)
    idx = rng.choice(len(train_smiles), size=10_000, replace=False)
    ref2_smiles = [train_smiles[i] for i in idx]
    print("building ref2 from a different 10k of training SMILES ...")
    t0 = time.time()
    chemnet = fcd.load_ref_model()
    A_ref2 = fcd.get_predictions(chemnet, ref2_smiles)
    mu2 = A_ref2.mean(0)
    cov2 = np.cov(A_ref2.T)
    print(f"  ref2 built in {time.time() - t0:.1f}s")

    # ref1 vs ref2 — pure noise floor between two independent 10k slices
    floor = float(
        fcd.calculate_frechet_distance(mu1=mu1, mu2=mu2, sigma1=cov1, sigma2=cov2)
    )
    print(f"\nref1 ↔ ref2 FCD = {floor:.4f}  (cross-reference noise floor)\n")

    # Evaluate each v4 prediction against both refs
    files = sorted((Path("Generation-Challenge") / "predictions" / "v4").glob("v4_*.txt"))
    print(f"{'file':<35} {'FCD vs ref1':>14} {'FCD vs ref2':>14}  ratio")
    print("-" * 80)
    for p in files:
        sub = read_lines(p)[:10_000]
        f1 = fcd_val(sub, mu1, cov1)
        f2 = fcd_val(sub, mu2, cov2)
        ratio = f2 / f1 if f1 > 0 else float("nan")
        print(f"{p.name:<35} {f1:>14.4f} {f2:>14.4f}  {ratio:>5.2f}")

    # Also the submitted v3 file as control
    v3_path = Path("Generation-Challenge") / "predictions" / "v3" / "v3_T1.0.txt"
    if v3_path.exists():
        sub = read_lines(v3_path)[:10_000]
        f1 = fcd_val(sub, mu1, cov1)
        f2 = fcd_val(sub, mu2, cov2)
        ratio = f2 / f1 if f1 > 0 else float("nan")
        print(f"{'v3_T1.0.txt (submitted)':<35} {f1:>14.4f} {f2:>14.4f}  {ratio:>5.2f}")


if __name__ == "__main__":
    main()
