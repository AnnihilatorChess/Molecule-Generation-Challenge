"""v6: Extend the cached v4 pool by sampling more from v3 until the total reaches
target_total. Computes ChemNet activations on the new SMILES only and concatenates
with the cached activations, so we don't redo the work for the existing 50k.

Output: artifacts/v6/pool.pkl  with {smiles: list[str], activations: np.ndarray[N, 512]}.
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
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v3"))

from common.paths import ARTIFACTS_DIR, EVAL_TRAIN_FILE, version_dirs  # noqa: E402
from common.postprocess import canonicalize, load_train_set  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402
from model import TransformerLM  # noqa: E402

VERSION = "v6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=str(ARTIFACTS_DIR / "v3" / "ckpt_best.pt")
    )
    parser.add_argument(
        "--existing_pool",
        type=str,
        default=str(ARTIFACTS_DIR / "v4" / "pool.pkl"),
        help="Pool to extend (its SMILES become the exclusion set + the first chunk of output).",
    )
    parser.add_argument("--target_total", type=int, default=200_000)
    parser.add_argument("--chunk_size", type=int, default=30_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, art_dir, _ = version_dirs(VERSION)
    out_path = art_dir / "pool.pkl"

    print(f"loading existing pool: {args.existing_pool}")
    with open(args.existing_pool, "rb") as f:
        existing = pickle.load(f)
    existing_smiles: list[str] = list(existing["smiles"])
    existing_acts: np.ndarray = existing["activations"]
    print(
        f"  existing pool: {len(existing_smiles):,} SMILES, "
        f"activations shape {existing_acts.shape}"
    )
    existing_set = set(existing_smiles)

    n_needed = args.target_total - len(existing_smiles)
    if n_needed <= 0:
        print(f"existing pool already at {len(existing_smiles):,} >= target. Nothing to do.")
        return
    print(f"need {n_needed:,} additional unique novel SMILES")

    if args.seed:
        torch.manual_seed(args.seed)

    print(f"loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = SmilesTokenizer(ckpt["tokenizer"])
    m_args = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(
        vocab_size=tok.vocab_size,
        max_len=m_args["max_len"],
        dim=m_args["dim"],
        n_heads=m_args["n_heads"],
        n_layers=m_args["n_layers"],
        mlp_ratio=m_args["mlp_ratio"],
        dropout=0.0,
        pad_id=tok.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"loaded v3 model: dim={m_args['dim']}, layers={m_args['n_layers']}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    print("loading eval-bundle train set ...")
    train_set = load_train_set(EVAL_TRAIN_FILE)
    print(f"  {len(train_set):,} train SMILES")

    new_smiles: list[str] = []
    new_set: set[str] = set()
    total_raw = 0
    total_valid = 0
    rounds = 0
    while len(new_smiles) < n_needed and rounds < args.max_rounds:
        rounds += 1
        chunk = args.chunk_size
        print(
            f"  round {rounds}: sampling {chunk:,} (new pool {len(new_smiles):,} / {n_needed:,})"
        )
        t0 = time.time()
        seqs = model.sample(
            n=chunk,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            pad_id=tok.pad_id,
            max_len=args.max_len,
            temperature=args.temperature,
            device=device,
            batch_size=args.batch,
        )
        total_raw += chunk
        decoded = [tok.decode(s) for s in seqs]
        decoded = [s for s in decoded if s]
        canon = canonicalize(decoded)
        valid = [c for c in canon if c is not None]
        total_valid += len(valid)
        added = 0
        for c in valid:
            if c in train_set or c in existing_set or c in new_set:
                continue
            new_set.add(c)
            new_smiles.append(c)
            added += 1
        print(
            f"    decoded={len(decoded)} valid={len(valid)} added={added} "
            f"new_pool={len(new_smiles)}   ({time.time() - t0:.1f}s)"
        )

    print(f"final new SMILES: {len(new_smiles):,} (raw sampled this run: {total_raw:,})")
    new_smiles = new_smiles[:n_needed]

    # ChemNet on the new SMILES only
    print("computing ChemNet activations on new SMILES ...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import fcd  # cpu

    chemnet = fcd.load_ref_model()
    t0 = time.time()
    new_acts = fcd.get_predictions(chemnet, new_smiles)
    print(f"  new activations shape = {new_acts.shape}   ({time.time() - t0:.1f}s)")

    # Concatenate
    all_smiles = existing_smiles + new_smiles
    all_acts = np.concatenate([existing_acts, new_acts.astype(existing_acts.dtype)], axis=0)
    print(f"final pool: {len(all_smiles):,} SMILES, activations shape {all_acts.shape}")

    with open(out_path, "wb") as f:
        pickle.dump({"smiles": all_smiles, "activations": all_acts}, f)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    summary = {
        "ckpt": str(args.ckpt),
        "existing_pool_path": str(args.existing_pool),
        "existing_pool_size": len(existing_smiles),
        "n_new": len(new_smiles),
        "final_pool_size": len(all_smiles),
        "rounds": rounds,
        "total_raw": total_raw,
        "total_valid_after_canon": total_valid,
        "temperature": args.temperature,
    }
    with open(art_dir / "extend_pool_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {art_dir / 'extend_pool_summary.json'}")


if __name__ == "__main__":
    main()
