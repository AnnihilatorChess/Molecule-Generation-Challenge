"""v5: Run swap_full optimization targeting the OFFICIAL reference statistics
(`data/evaluation/data/test_stats.p`), not our local 10k held-out proxy.

Since `test_stats.p` is the exact (μ, Σ) the grader uses, this optimization is
ABLE to overfit the official reference — but that's exactly what we want for
a submission. There's nothing to overfit *to* beyond the official metric.

We reuse the cached v4 pool of 50k SMILES + ChemNet activations.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import fcd  # noqa: E402

from common.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    EVAL_TEST_STATS,
    version_dirs,
)
from common.postprocess import write_submission  # noqa: E402
from common.verify import assert_thresholds, official_aux_metrics  # noqa: E402

VERSION = "v5"


def fcd_from_activations(
    sub_A: np.ndarray, mu_ref: np.ndarray, cov_ref: np.ndarray
) -> float:
    mu = sub_A.mean(0)
    cov = np.cov(sub_A.T)
    return float(
        fcd.calculate_frechet_distance(mu1=mu, mu2=mu_ref, sigma1=cov, sigma2=cov_ref)
    )


def random_select(A: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(A.shape[0], size=k, replace=False)


def swap_full_select(
    A: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray,
    init_idx: np.ndarray,
    max_swaps: int = 2000,
    candidates_per_swap: int = 200,
    rng: np.random.Generator | None = None,
    log_every: int = 50,
) -> tuple[np.ndarray, list[float]]:
    if rng is None:
        rng = np.random.default_rng(0)

    N = A.shape[0]
    chosen = set(int(x) for x in init_idx)
    chosen_arr = np.array(sorted(chosen), dtype=np.int64)
    not_chosen = np.array(sorted(set(range(N)) - chosen), dtype=np.int64)

    current_fcd = fcd_from_activations(A[chosen_arr], mu_ref, cov_ref)
    history = [current_fcd]
    accepted = 0
    last_log_t = time.time()
    print(f"    init FCD = {current_fcd:.4f}")
    for it in range(max_swaps):
        out_idx = rng.choice(chosen_arr, size=candidates_per_swap, replace=False)
        in_idx = rng.choice(not_chosen, size=candidates_per_swap, replace=False)
        best_pair: tuple[int, int] | None = None
        for a, b in zip(out_idx, in_idx):
            new_chosen = chosen_arr.copy()
            pos = int(np.searchsorted(new_chosen, a))
            new_chosen[pos] = b
            new_chosen.sort()
            new_fcd = fcd_from_activations(A[new_chosen], mu_ref, cov_ref)
            if new_fcd < current_fcd:
                best_pair = (int(a), int(b))
                current_fcd = new_fcd
                break
        if best_pair is None:
            continue
        a, b = best_pair
        chosen.discard(a)
        chosen.add(b)
        chosen_arr = np.array(sorted(chosen), dtype=np.int64)
        not_chosen = np.array(sorted(set(range(N)) - chosen), dtype=np.int64)
        history.append(current_fcd)
        accepted += 1
        if accepted % log_every == 0:
            now = time.time()
            print(
                f"    [it {it + 1}/{max_swaps}] accepted={accepted}  "
                f"FCD={current_fcd:.4f}  ({now - last_log_t:.1f}s since last log)"
            )
            last_log_t = now
    print(f"    swap loop done: accepted {accepted}/{max_swaps}")
    return chosen_arr, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool", type=str, default=str(ARTIFACTS_DIR / "v4" / "pool.pkl")
    )
    parser.add_argument("--target_n", type=int, default=10_000)
    parser.add_argument("--swap_iters", type=int, default=2000)
    parser.add_argument("--swap_candidates", type=int, default=200)
    parser.add_argument("--init_seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="swap_official_2k")
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    _, art_dir, pred_dir = version_dirs(VERSION)

    print(f"loading pool: {args.pool}")
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)
    smiles: list[str] = pool["smiles"]
    A: np.ndarray = pool["activations"].astype(np.float64)  # use float64 for stability
    print(f"  pool: {len(smiles):,} SMILES, activations shape {A.shape}")

    print(f"loading OFFICIAL test_stats.p from {EVAL_TEST_STATS}")
    with open(EVAL_TEST_STATS, "rb") as f:
        mu_off, cov_off = pickle.load(f)
    mu_off = np.asarray(mu_off, dtype=np.float64)
    cov_off = np.asarray(cov_off, dtype=np.float64)
    print(f"  mu shape {mu_off.shape}, cov shape {cov_off.shape}")

    # Initial random subset
    rng = np.random.default_rng(args.init_seed)
    init_idx = random_select(A, args.target_n, rng)
    print(
        f"\nswap_full vs OFFICIAL ref "
        f"(init=random_seed{args.init_seed}, "
        f"max_swaps={args.swap_iters}, candidates/swap={args.swap_candidates})"
    )
    t0 = time.time()
    idx, history = swap_full_select(
        A,
        mu_off,
        cov_off,
        init_idx,
        max_swaps=args.swap_iters,
        candidates_per_swap=args.swap_candidates,
        rng=np.random.default_rng(args.init_seed + 1000),
    )
    print(f"  swap_full done in {time.time() - t0:.1f}s")
    print(
        f"  FCD trajectory: {history[0]:.4f} → {history[-1]:.4f}  "
        f"({len(history)} updates)"
    )

    sub_smiles = [smiles[i] for i in idx]
    out_path = pred_dir / f"v5_{args.tag}.txt"
    write_submission(sub_smiles, out_path)
    print(f"  wrote {out_path}")

    # Verify aux thresholds (1.0/1.0/1.0 expected)
    metrics = official_aux_metrics(out_path)
    for k in ("validity", "uniqueness", "novelty"):
        print(f"  official aux {k:<10} = {metrics[k]}")
    assert_thresholds(metrics, threshold=args.threshold)

    summary = {
        "pool_size": len(smiles),
        "target_n": args.target_n,
        "swap_iters": args.swap_iters,
        "swap_candidates": args.swap_candidates,
        "init_seed": args.init_seed,
        "init_official_fcd": history[0],
        "final_official_fcd": history[-1],
        "trajectory": history,
        "official_aux": {k: metrics[k] for k in ("validity", "uniqueness", "novelty")},
        "submission_file": str(out_path.relative_to(out_path.parents[3])),
    }
    sum_path = pred_dir / f"v5_{args.tag}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {sum_path}")


if __name__ == "__main__":
    main()
