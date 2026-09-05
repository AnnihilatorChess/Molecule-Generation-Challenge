"""v5 step 1: Load the eval bundle's `test_stats.p` (which IS the official
reference) and verify that recomputing FCD against it locally reproduces the
server-reported numbers for our previous submissions.

If this passes (v2 ≈ 0.359, v3 ≈ 0.258) we have proven that we can compute
the exact official FCD ourselves — no more proxy, no more calibration ratio.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import fcd  # noqa: E402

from common.local_fcd import compute_stats  # noqa: E402
from common.paths import CHALLENGE_ROOT, EVAL_TEST_STATS  # noqa: E402


def read_lines(p: Path) -> list[str]:
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main() -> None:
    print(f"loading official test_stats.p from {EVAL_TEST_STATS}")
    with open(EVAL_TEST_STATS, "rb") as f:
        mu_off, cov_off = pickle.load(f)
    print(
        f"  official mu shape {mu_off.shape}, cov shape {cov_off.shape}, "
        f"dtype mu={mu_off.dtype} cov={cov_off.dtype}"
    )

    chemnet = fcd.load_ref_model()
    print("ChemNet loaded.\n")

    targets: list[tuple[str, Path, float | None]] = [
        ("v1_T1.0", CHALLENGE_ROOT / "predictions" / "v1" / "v1_T1.0_verified.txt", None),
        ("v2_T1.0 (submitted #1)", CHALLENGE_ROOT / "predictions" / "v2" / "v2_T1.0.txt", 0.359),
        ("v3_T1.0 (submitted #2)", CHALLENGE_ROOT / "predictions" / "v3" / "v3_T1.0.txt", 0.258),
        ("v4_random_seed0", CHALLENGE_ROOT / "predictions" / "v4" / "v4_random_seed0.txt", None),
        ("v4_random_seed1", CHALLENGE_ROOT / "predictions" / "v4" / "v4_random_seed1.txt", None),
        ("v4_random_seed2", CHALLENGE_ROOT / "predictions" / "v4" / "v4_random_seed2.txt", None),
        ("v4_greedy_mean", CHALLENGE_ROOT / "predictions" / "v4" / "v4_greedy_mean.txt", None),
        ("v4_swap_full", CHALLENGE_ROOT / "predictions" / "v4" / "v4_swap_full.txt", None),
    ]

    print(f"{'file':<28} {'FCD vs official':>18} {'server':>10} {'delta':>10}")
    print("-" * 70)
    for label, path, server in targets:
        if not path.exists():
            print(f"{label:<28} {'MISSING':>18}")
            continue
        sub = read_lines(path)[:10_000]
        mu, cov = compute_stats(sub, model=chemnet)
        f_off = float(
            fcd.calculate_frechet_distance(
                mu1=mu, mu2=mu_off, sigma1=cov, sigma2=cov_off
            )
        )
        if server is None:
            print(f"{label:<28} {f_off:>18.4f}")
        else:
            delta = f_off - server
            print(f"{label:<28} {f_off:>18.4f} {server:>10.4f} {delta:>10.4f}")


if __name__ == "__main__":
    main()
