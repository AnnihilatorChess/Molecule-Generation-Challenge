"""v4 step 2: subset-selection methods on the cached pool.

Loads `artifacts/v4/pool.pkl` (smiles + ChemNet activations), tries several
methods of picking 10,000 indices, and reports local FCD for each. Writes one
submission file per method.

Methods:
  - random_<seed>       : random 10k from the pool (baseline)
  - greedy_mean         : iteratively add the candidate that minimizes
                          ||μ_subset − μ_ref||² (mean-only FCD proxy)
  - swap_full           : start from a random 10k, then run greedy 1-for-1 swaps
                          that strictly decrease the full FCD. Bounded by
                          --swap_iters.
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

from common.local_fcd import load_ref  # noqa: E402
from common.paths import ARTIFACTS_DIR, version_dirs  # noqa: E402
from common.postprocess import write_submission  # noqa: E402
from common.verify import assert_thresholds, official_aux_metrics  # noqa: E402

VERSION = "v4"


def fcd_from_activations(
    sub_A: np.ndarray, mu_ref: np.ndarray, cov_ref: np.ndarray
) -> float:
    mu = sub_A.mean(0)
    cov = np.cov(sub_A.T)
    return float(
        fcd.calculate_frechet_distance(mu1=mu, mu2=mu_ref, sigma1=cov, sigma2=cov_ref)
    )


def random_select(
    A: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.choice(A.shape[0], size=k, replace=False)


def greedy_mean_select(A: np.ndarray, mu_ref: np.ndarray, k: int) -> np.ndarray:
    """At each step pick the candidate that minimizes ||new_mean − μ_ref||².

    `new_mean` after adding candidate i to current set of size n:
        new_mean = (n*current_mean + A[i]) / (n+1)
    Minimizing ||new_mean − μ_ref||² is equivalent to minimizing
        ||A[i] − ((n+1) μ_ref − n * current_mean)||²
    so we just nearest-neighbour against a moving target.
    """
    N, D = A.shape
    chosen = np.zeros(k, dtype=np.int64)
    chosen_mask = np.zeros(N, dtype=bool)
    sum_a = np.zeros(D, dtype=np.float64)
    for step in range(k):
        target = (step + 1) * mu_ref - sum_a  # shape (D,)
        # squared dist of every row to target
        diffs = A - target
        dists = np.einsum("ij,ij->i", diffs, diffs)
        dists[chosen_mask] = np.inf
        i_pick = int(np.argmin(dists))
        chosen[step] = i_pick
        chosen_mask[i_pick] = True
        sum_a += A[i_pick]
    return chosen


def swap_full_select(
    A: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray,
    init_idx: np.ndarray,
    max_swaps: int = 200,
    candidates_per_swap: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Greedy 1-for-1 swap optimization of the full FCD.

    Each iteration picks (one random in-subset, one random out-of-subset) and
    accepts the swap if it strictly decreases FCD. Repeats `max_swaps` times.
    Returns the resulting indices and the FCD trajectory.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = A.shape[0]
    chosen = set(int(x) for x in init_idx)
    chosen_arr = np.array(sorted(chosen), dtype=np.int64)
    not_chosen = np.array(sorted(set(range(N)) - chosen), dtype=np.int64)

    current_fcd = fcd_from_activations(A[chosen_arr], mu_ref, cov_ref)
    history = [current_fcd]
    accepted = 0
    for it in range(max_swaps):
        # consider a batch of candidate swaps
        out_idx = rng.choice(chosen_arr, size=candidates_per_swap, replace=False)
        in_idx = rng.choice(not_chosen, size=candidates_per_swap, replace=False)
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        # try each pair (cheap candidates first)
        for a, b in zip(out_idx, in_idx):
            new_chosen = chosen_arr.copy()
            # find position of a in chosen_arr
            pos = int(np.searchsorted(new_chosen, a))
            new_chosen[pos] = b
            new_chosen.sort()
            new_fcd = fcd_from_activations(A[new_chosen], mu_ref, cov_ref)
            delta = new_fcd - current_fcd
            if delta < best_delta:
                best_delta = delta
                best_pair = (int(a), int(b))
                # break on any improvement (greedy first-fit)
                break
        if best_pair is None:
            continue
        a, b = best_pair
        chosen.discard(a)
        chosen.add(b)
        chosen_arr = np.array(sorted(chosen), dtype=np.int64)
        not_chosen = np.array(sorted(set(range(N)) - chosen), dtype=np.int64)
        current_fcd = fcd_from_activations(A[chosen_arr], mu_ref, cov_ref)
        history.append(current_fcd)
        accepted += 1
    print(f"    swap loop: accepted {accepted}/{max_swaps}")
    return chosen_arr, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, default=str(ARTIFACTS_DIR / "v4" / "pool.pkl"))
    parser.add_argument("--ref", type=str, default=str(ARTIFACTS_DIR / "local_fcd_ref.pkl"))
    parser.add_argument("--target_n", type=int, default=10_000)
    parser.add_argument("--swap_iters", type=int, default=200)
    parser.add_argument("--swap_candidates", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["random", "greedy_mean", "swap_full"],
        help="which methods to run",
    )
    args = parser.parse_args()

    _, art_dir, pred_dir = version_dirs(VERSION)

    print(f"loading pool: {args.pool}")
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)
    smiles: list[str] = pool["smiles"]
    A: np.ndarray = pool["activations"]
    print(f"  pool: {len(smiles):,} SMILES, activations shape {A.shape}")

    print(f"loading local ref: {args.ref}")
    mu_ref, cov_ref = load_ref(Path(args.ref))
    print(f"  mu_ref shape {mu_ref.shape}, cov_ref shape {cov_ref.shape}")

    results: list[dict] = []

    def evaluate_and_write(idx: np.ndarray, name: str) -> dict:
        assert len(idx) == args.target_n, f"need {args.target_n}, got {len(idx)}"
        sub_smiles = [smiles[i] for i in idx]
        sub_A = A[idx]
        local = fcd_from_activations(sub_A, mu_ref, cov_ref)
        out = pred_dir / f"v4_{name}.txt"
        write_submission(sub_smiles, out)
        metrics = official_aux_metrics(out)
        assert_thresholds(metrics, threshold=args.threshold)
        rec = {
            "method": name,
            "file": str(out.relative_to(out.parents[3])),
            "local_fcd": local,
            "official_aux": {k: metrics[k] for k in ("validity", "uniqueness", "novelty")},
        }
        print(f"  {name:>22}  local FCD = {local:.4f}   "
              f"aux = {metrics['validity']:.4f}/{metrics['uniqueness']:.4f}/{metrics['novelty']:.4f}")
        results.append(rec)
        return rec

    # ---- baseline: random selections ----
    if "random" in args.methods:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = random_select(A, args.target_n, rng)
            evaluate_and_write(idx, f"random_seed{seed}")

    # ---- greedy mean-only ----
    if "greedy_mean" in args.methods:
        print("greedy mean-only selection ...")
        t0 = time.time()
        idx = greedy_mean_select(A, mu_ref, args.target_n)
        print(f"  greedy_mean done in {time.time() - t0:.1f}s")
        evaluate_and_write(idx, "greedy_mean")

    # ---- swap_full ----
    if "swap_full" in args.methods:
        # init from best random seed (use seed 0 baseline)
        init_seed = args.seeds[0]
        rng = np.random.default_rng(init_seed)
        init_idx = random_select(A, args.target_n, rng)
        print(
            f"swap_full local search "
            f"(init=random_seed{init_seed}, max_swaps={args.swap_iters}, "
            f"candidates/swap={args.swap_candidates}) ..."
        )
        t0 = time.time()
        idx, history = swap_full_select(
            A, mu_ref, cov_ref, init_idx,
            max_swaps=args.swap_iters,
            candidates_per_swap=args.swap_candidates,
            rng=np.random.default_rng(init_seed + 1000),
        )
        print(
            f"  swap_full done in {time.time() - t0:.1f}s  "
            f"(FCD trajectory: {history[0]:.4f} → {history[-1]:.4f}, "
            f"n_history={len(history)})"
        )
        rec = evaluate_and_write(idx, "swap_full")
        rec["swap_history"] = history

    # sort by local FCD and print summary
    results_sorted = sorted(results, key=lambda r: r["local_fcd"])
    print("\nSummary (sorted by local FCD):")
    for r in results_sorted:
        print(f"  {r['method']:>22}  {r['local_fcd']:.4f}   {r['file']}")

    summary = {
        "pool_size": len(smiles),
        "results_sorted": results_sorted,
    }
    with open(art_dir / "select_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {art_dir / 'select_summary.json'}")


if __name__ == "__main__":
    main()
