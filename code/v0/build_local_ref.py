"""v0: Build the local FCD reference.

- Reads the full training corpus.
- Sets aside a fixed held-out slice of 10,000 SMILES (seed=1234).
- Writes the held-out SMILES and the remaining training SMILES.
- Computes ChemNet (mean, cov) on the held-out slice and pickles it.

This held-out slice is our private FCD test set. Models will be trained on the
remaining ~1,262,851 SMILES and selected by FCD against this local reference.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # add code/ to path

from common.local_fcd import compute_stats, save_ref  # noqa: E402
from common.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    TRAIN_FILE,
    version_dirs,
)

HELDOUT_N = 10_000
SEED = 1234
VERSION = "v0"


def main() -> None:
    _, art_dir, _ = version_dirs(VERSION)

    print(f"Reading {TRAIN_FILE} ...")
    with open(TRAIN_FILE) as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    n_total = len(all_smiles)
    print(f"  {n_total:,} SMILES total")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n_total)
    heldout_idx = sorted(idx[:HELDOUT_N].tolist())
    train_idx = sorted(idx[HELDOUT_N:].tolist())

    heldout_smiles = [all_smiles[i] for i in heldout_idx]
    train_smiles = [all_smiles[i] for i in train_idx]
    print(f"  held out: {len(heldout_smiles):,}")
    print(f"  training: {len(train_smiles):,}")

    heldout_path = art_dir / "heldout_smiles.txt"
    train_path = art_dir / "train_smiles.txt"
    with open(heldout_path, "w", newline="\n") as f:
        f.write("\n".join(heldout_smiles) + "\n")
    with open(train_path, "w", newline="\n") as f:
        f.write("\n".join(train_smiles) + "\n")
    print(f"  wrote {heldout_path}")
    print(f"  wrote {train_path}")

    print("Loading ChemNet (CPU) and computing (mean, cov) on held-out slice ...")
    t0 = time.time()
    mu, cov = compute_stats(heldout_smiles)
    print(
        f"  done in {time.time() - t0:.1f}s -- mu shape {mu.shape}, cov shape {cov.shape}"
    )

    ref_path = art_dir / "local_fcd_ref.pkl"
    save_ref(mu, cov, ref_path)
    print(f"  wrote {ref_path}")

    # Also drop a shared symlink-style copy at artifacts/local_fcd_ref.pkl for convenience
    shared_path = ARTIFACTS_DIR / "local_fcd_ref.pkl"
    save_ref(mu, cov, shared_path)
    print(f"  wrote {shared_path}")


if __name__ == "__main__":
    main()
