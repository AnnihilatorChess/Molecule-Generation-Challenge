"""v0: End-to-end smoke test of the submission pipeline.

We do NOT submit — we only validate locally that:

1. The post-processing pipeline runs and produces a 10,000-line file.
2. Our local FCD harness gives sensible numbers.
3. We can reproduce the official metric numbers (via metric_fcd.py).

Two probes:

A) Probe 1: the held-out slice (10,000 RDKit-canonical SMILES that ARE in train).
   - Against the *eval bundle's* train set, novelty = 0 (they're in there).
   - Local FCD vs the local reference should be ≈ 0 (same set).
   - This is the validity/FCD floor and a tight sanity check.

B) Probe 2: take 10,000 SMILES from the training corpus that are NOT in the held-out
   slice. Their canonical strings are in the eval-bundle train set → novelty = 0 again,
   but FCD will be near zero. This confirms the FCD-only floor.

Neither probe is a valid submission (novelty fails); both are diagnostic.

Optionally, we also call the official `metric_fcd.py` on `sample_submission.txt`
to confirm the official evaluator runs on this machine.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # add code/ to path

from common.local_fcd import _load_model, local_fcd  # noqa: E402
from common.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    CHALLENGE_ROOT,
    EVAL_TRAIN_FILE,
    SAMPLE_SUBMISSION,
    version_dirs,
)
from common.postprocess import (  # noqa: E402
    canonicalize,
    load_train_set,
    postprocess,
    write_submission,
)


VERSION = "v0"


def read_lines(p: Path) -> list[str]:
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main() -> None:
    _, art_dir, pred_dir = version_dirs(VERSION)
    ref_path = ARTIFACTS_DIR / "local_fcd_ref.pkl"
    heldout_path = art_dir / "heldout_smiles.txt"
    train_path = art_dir / "train_smiles.txt"

    for required in [ref_path, heldout_path, train_path]:
        if not required.exists():
            print(f"MISSING: {required}\nRun build_local_ref.py first.", file=sys.stderr)
            sys.exit(1)

    print("Loading eval-bundle train set (novelty reference)...")
    eval_train = load_train_set(EVAL_TRAIN_FILE)
    print(f"  {len(eval_train):,} unique SMILES in eval-bundle train file")

    # ------------------------------------------------------------------
    # Probe 1: the held-out slice itself.
    #   These ARE in eval_train (the bundle ships the FULL training set).
    #   So we expect: validity=1.0, uniqueness=1.0, novelty=0.0, local FCD ~ 0.
    # ------------------------------------------------------------------
    print("\nProbe 1: held-out slice (sanity floor) ----------------------------")
    heldout = read_lines(heldout_path)
    print(f"  {len(heldout):,} SMILES loaded")
    canon = canonicalize(heldout)
    valid = [s for s in canon if s is not None]
    unique = list(dict.fromkeys(valid))
    in_train = sum(1 for s in unique if s in eval_train)
    print(
        f"  validity  = {len(valid) / len(heldout):.4f}\n"
        f"  uniqueness= {len(unique) / len(heldout):.4f}\n"
        f"  in-eval-train: {in_train}/{len(unique)}"
    )
    chemnet = _load_model()
    t0 = time.time()
    fcd_val, _ = local_fcd(unique[:10_000], ref_path, model=chemnet)
    print(
        f"  local FCD (heldout vs heldout reference): {fcd_val:.4f}   "
        f"({time.time() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------
    # Probe 2: take 10k SMILES from the *non-heldout* training slice.
    #   They are still in eval_train, so novelty = 0 — but they're disjoint from
    #   our local FCD reference set, giving us a non-trivial in-distribution FCD.
    # ------------------------------------------------------------------
    print("\nProbe 2: 10k training (non-heldout) samples vs local ref ----------")
    train_other = read_lines(train_path)
    probe2 = train_other[:10_000]
    canon2 = canonicalize(probe2)
    valid2 = [s for s in canon2 if s is not None]
    unique2 = list(dict.fromkeys(valid2))
    t0 = time.time()
    fcd2, _ = local_fcd(unique2[:10_000], ref_path, model=chemnet)
    print(
        f"  validity   = {len(valid2) / len(probe2):.4f}\n"
        f"  uniqueness = {len(unique2) / len(probe2):.4f}\n"
        f"  local FCD  = {fcd2:.4f}   ({time.time() - t0:.1f}s)"
    )
    print(
        "  ^ This is the in-distribution FCD floor: even 10k *real* training molecules\n"
        "    won't give FCD=0 against a different 10k slice. Our generated submissions\n"
        "    should aim to land near this number."
    )

    # ------------------------------------------------------------------
    # Probe 3: post-process the canonical training samples into a *would-be*
    #   submission file (will not actually pass novelty, but tests the writer).
    # ------------------------------------------------------------------
    print("\nProbe 3: post-process pipeline test --------------------------------")
    try:
        sub, stats = postprocess(probe2, eval_train, target_n=10_000)
        print(f"  unexpectedly got a novel set: stats={stats}")
    except ValueError as e:
        print(f"  expected failure (training slice has no novelty): {e}")

    # Smoke test the writer with a *fake* novel set: append the held-out slice
    # canonical strings to themselves with a random tail atom so they become novel.
    # This is just to test the file format, NOT to submit.
    fake_novel = unique[:10_000]  # we *will* falsely treat these as novel for writer test
    out_path = pred_dir / "smoke_format.txt"
    write_submission(fake_novel, out_path)
    print(f"  wrote dummy submission file {out_path} ({out_path.stat().st_size} bytes)")

    # ------------------------------------------------------------------
    # Probe 4: run the OFFICIAL evaluator on sample_submission.txt.
    #   - sample_submission.txt has 9,999 SMILES, all (presumably) from train.
    #   - This proves metric_fcd.py works end-to-end on this machine.
    # ------------------------------------------------------------------
    print("\nProbe 4: official metric_fcd.py on sample_submission.txt ----------")
    eval_tar = CHALLENGE_ROOT / "data" / "evaluation.tar"
    if not eval_tar.exists():
        print(
            "  data/evaluation.tar not present (already unpacked); "
            "calling get_metric directly via Python instead."
        )
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        sys.path.insert(0, str(CHALLENGE_ROOT / "data"))
        from evaluation.evaluate_submission import get_metric  # type: ignore

        class Args:
            submission = str(SAMPLE_SUBMISSION)
            trainset = str(EVAL_TRAIN_FILE)
            teststats = str(CHALLENGE_ROOT / "data" / "evaluation" / "data" / "test_stats.p")

        args = Args()
        for name in ["validity", "uniqueness", "novelty"]:
            v = get_metric(args, name)
            print(f"  official {name:<10} = {v}")
        t0 = time.time()
        v = get_metric(args, "fcd")
        print(f"  official fcd        = {v}   ({time.time() - t0:.1f}s)")
    else:
        cmd = [
            sys.executable,
            str(CHALLENGE_ROOT / "metric_fcd.py"),
            "--submission",
            str(SAMPLE_SUBMISSION),
            "--target",
            str(eval_tar),
        ]
        print("  $ " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nSmoke OK.")


if __name__ == "__main__":
    main()
