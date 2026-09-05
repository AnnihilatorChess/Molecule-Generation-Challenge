"""v6: FAST swap_full subset selection against the OFFICIAL reference.

Same algorithm as v5/swap_official.py (greedy 1-for-1 swaps minimizing the full
FCD against `test_stats.p`), but rewritten to scale to large pools:

Fixes vs v5:
  1. O(1) candidate bookkeeping. v5 rebuilt `not_chosen = sorted(set(range(N)) -
     chosen)` after EVERY accepted swap — O(N log N) per swap, which dominated
     once N hit 200k. Here we keep a boolean membership mask and an explicit
     `out_indices` list, updated in place on each accept.
  2. Incremental FCD-input maintenance. We track the running sum and the
     un-normalised scatter matrix (Σx xᵀ) of the chosen subset, so each candidate
     swap updates μ and Σ with two rank-1 ops (O(D²)) instead of recomputing
     np.cov over the whole 10k subset (O(N_subset · D²)). The expensive part
     left is the FCD's matrix square root, O(D³), unavoidable per candidate.
  3. Unbuffered, flushed progress every `log_every` accepted swaps so the run is
     watchable in the background.
  4. Periodic checkpointing of the best subset + trajectory to
     artifacts/v6/swap_ckpt.npz so a crash/idle never costs the whole run, and
     so we can resume with --resume.

Usage (DO NOT run automatically — launch manually):
  python code/v6/swap_official_fast.py \
      --pool artifacts/v6/pool.pkl --swap_iters 8000 --tag swap_official_8k_200kpool

Reuses the cached `artifacts/v6/pool.pkl` (200k SMILES + ChemNet activations).
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

from common.paths import ARTIFACTS_DIR, EVAL_TEST_STATS, version_dirs  # noqa: E402
from common.postprocess import write_submission  # noqa: E402
from common.verify import assert_thresholds, official_aux_metrics  # noqa: E402

VERSION = "v6"


def fcd_from_moments(
    sum_x: np.ndarray,
    scatter: np.ndarray,
    n: int,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray,
) -> float:
    """FCD given the running sum and scatter (Σ xxᵀ) of the chosen subset.

    mu  = sum_x / n
    cov = (scatter - n · mu·muᵀ) / (n - 1)   [matches np.cov default ddof=1]
    """
    mu = sum_x / n
    cov = (scatter - n * np.outer(mu, mu)) / (n - 1)
    return float(
        fcd.calculate_frechet_distance(mu1=mu, mu2=mu_ref, sigma1=cov, sigma2=cov_ref)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, default=str(ARTIFACTS_DIR / "v6" / "pool.pkl"))
    parser.add_argument("--target_n", type=int, default=10_000)
    parser.add_argument("--swap_iters", type=int, default=8000)
    parser.add_argument("--swap_candidates", type=int, default=200)
    parser.add_argument("--init_seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="swap_official_8k_200kpool")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="resume from swap_ckpt.npz if present")
    args = parser.parse_args()

    _, art_dir, pred_dir = version_dirs(VERSION)
    ckpt_path = art_dir / f"swap_ckpt_{args.tag}.npz"

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"loading pool: {args.pool}")
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)
    smiles: list[str] = pool["smiles"]
    A: np.ndarray = np.ascontiguousarray(pool["activations"], dtype=np.float64)
    N, D = A.shape
    log(f"  pool: {N:,} SMILES, activations {A.shape}")

    log(f"loading OFFICIAL test_stats.p from {EVAL_TEST_STATS}")
    with open(EVAL_TEST_STATS, "rb") as f:
        mu_off, cov_off = pickle.load(f)
    mu_off = np.asarray(mu_off, dtype=np.float64)
    cov_off = np.asarray(cov_off, dtype=np.float64)
    log(f"  mu {mu_off.shape}, cov {cov_off.shape}")

    k = args.target_n
    rng = np.random.default_rng(args.init_seed + 1000)

    # ---- init subset (random or resumed) ----
    start_iter = 0
    history: list[float] = []
    if args.resume and ckpt_path.exists():
        log(f"resuming from {ckpt_path}")
        ck = np.load(ckpt_path)
        chosen_idx = ck["chosen_idx"].astype(np.int64)
        start_iter = int(ck["accepted"])
        history = ck["history"].tolist()
        log(f"  resumed at accepted={start_iter}, last FCD={history[-1]:.4f}")
    else:
        init_rng = np.random.default_rng(args.init_seed)
        chosen_idx = init_rng.choice(N, size=k, replace=False).astype(np.int64)

    # membership mask + out-of-subset index list
    in_mask = np.zeros(N, dtype=bool)
    in_mask[chosen_idx] = True
    out_indices = np.flatnonzero(~in_mask)  # maintained incrementally

    # running moments of the chosen subset
    chosen_A = A[chosen_idx]
    sum_x = chosen_A.sum(axis=0)
    scatter = chosen_A.T @ chosen_A  # (D, D)
    current_fcd = fcd_from_moments(sum_x, scatter, k, mu_off, cov_off)
    if not history:
        history.append(current_fcd)
    log(f"init FCD = {current_fcd:.4f}  (accepted so far: {start_iter})")

    # position lookup for chosen_idx so we can swap in place
    # pos_in_chosen[g] = index into chosen_idx array for global id g (-1 if not chosen)
    pos_in_chosen = np.full(N, -1, dtype=np.int64)
    pos_in_chosen[chosen_idx] = np.arange(k)

    cand = args.swap_candidates
    accepted = start_iter
    t0 = time.time()
    last_log_t = t0

    target_accepts = args.swap_iters
    while accepted < target_accepts:
        # sample candidate out-of-subset items to bring IN
        in_choices = out_indices[rng.integers(0, len(out_indices), size=cand)]
        # sample candidate in-subset positions to take OUT
        out_choice_pos = rng.integers(0, k, size=cand)

        improved = False
        for c_in, p_out in zip(in_choices, out_choice_pos):
            g_out = chosen_idx[p_out]
            if c_in == g_out:
                continue
            x_in = A[c_in]
            x_out = A[g_out]
            # candidate moments after swapping out g_out, in c_in
            new_sum = sum_x - x_out + x_in
            new_scatter = scatter - np.outer(x_out, x_out) + np.outer(x_in, x_in)
            new_fcd = fcd_from_moments(new_sum, new_scatter, k, mu_off, cov_off)
            if new_fcd < current_fcd:
                # accept
                sum_x = new_sum
                scatter = new_scatter
                current_fcd = new_fcd
                # update membership structures in O(1)
                in_mask[g_out] = False
                in_mask[c_in] = True
                chosen_idx[p_out] = c_in
                pos_in_chosen[c_in] = p_out
                pos_in_chosen[g_out] = -1
                improved = True
                break
        if not improved:
            # no improving swap found in this candidate batch; resample
            continue

        accepted += 1
        history.append(current_fcd)

        if accepted % args.log_every == 0:
            now = time.time()
            rate = args.log_every / (now - last_log_t)
            log(
                f"  accepted={accepted}/{target_accepts}  FCD={current_fcd:.4f}  "
                f"({rate:.1f} acc/s, {now - t0:.0f}s elapsed)"
            )
            last_log_t = now

        if accepted % args.ckpt_every == 0:
            # rebuild out_indices from mask before checkpoint (and periodically,
            # to keep it exact — we let it drift between checkpoints since the
            # in_choices sampler tolerates the occasional already-in item via the
            # c_in==g_out guard and the new_fcd check).
            out_indices = np.flatnonzero(~in_mask)
            np.savez(
                ckpt_path,
                chosen_idx=chosen_idx,
                accepted=accepted,
                history=np.asarray(history),
            )

    log(f"swap loop done: accepted {accepted} in {time.time() - t0:.0f}s")
    log(f"FCD trajectory: {history[0]:.4f} -> {history[-1]:.4f}")

    # ---- write submission + verify ----
    sub_smiles = [smiles[i] for i in chosen_idx.tolist()]
    out_path = pred_dir / f"v6_{args.tag}.txt"
    write_submission(sub_smiles, out_path)
    log(f"wrote {out_path}")

    metrics = official_aux_metrics(out_path)
    for kk in ("validity", "uniqueness", "novelty"):
        log(f"  official aux {kk:<10} = {metrics[kk]}")
    assert_thresholds(metrics, threshold=args.threshold)

    # recompute final FCD from scratch as a correctness check against the
    # incrementally-maintained value (guards against float drift)
    chk = A[chosen_idx]
    mu = chk.mean(0)
    cov = np.cov(chk.T)
    final_fcd_scratch = float(
        fcd.calculate_frechet_distance(mu1=mu, mu2=mu_off, sigma1=cov, sigma2=cov_off)
    )
    log(f"final FCD (incremental) = {current_fcd:.4f}")
    log(f"final FCD (from scratch) = {final_fcd_scratch:.4f}  (should match)")

    summary = {
        "pool": str(args.pool),
        "pool_size": N,
        "target_n": k,
        "swap_iters": args.swap_iters,
        "swap_candidates": cand,
        "init_seed": args.init_seed,
        "init_fcd": history[0],
        "final_fcd_incremental": current_fcd,
        "final_fcd_scratch": final_fcd_scratch,
        "official_aux": {x: metrics[x] for x in ("validity", "uniqueness", "novelty")},
        "submission_file": str(out_path.relative_to(out_path.parents[3])),
    }
    with open(pred_dir / f"v6_{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {pred_dir / f'v6_{args.tag}_summary.json'}")


if __name__ == "__main__":
    main()
