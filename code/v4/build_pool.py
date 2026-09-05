"""v4 step 1: build a large canonical-unique-novel pool from the v3 checkpoint
and compute ChemNet activations on every member.

Output: artifacts/v4/pool.pkl  with {smiles: list[str], activations: np.ndarray[N, 512]}.

This file caches the expensive bits (sampling + ChemNet inference) so that
subset-selection experiments can iterate cheaply on the same pool.
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES_FCD", "-1")  # noop, just a hint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v3"))  # import the v3 model class

from common.paths import ARTIFACTS_DIR, EVAL_TRAIN_FILE, version_dirs  # noqa: E402
from common.postprocess import canonicalize, load_train_set  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402
from model import TransformerLM  # noqa: E402

VERSION = "v4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(ARTIFACTS_DIR / "v3" / "ckpt_best.pt"),
        help="path to v3 checkpoint",
    )
    parser.add_argument("--target_pool", type=int, default=50_000)
    parser.add_argument("--chunk_size", type=int, default=15_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max_rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, art_dir, _ = version_dirs(VERSION)
    pool_path = art_dir / "pool.pkl"
    print(f"output: {pool_path}")

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

    print(
        f"sampling until pool >= {args.target_pool:,} valid+unique+novel "
        f"(chunk={args.chunk_size}, T={args.temperature}, batch={args.batch})"
    )
    pool_set: set[str] = set()
    pool: list[str] = []
    total_raw = 0
    total_valid = 0
    rounds = 0
    while len(pool) < args.target_pool and rounds < args.max_rounds:
        rounds += 1
        print(
            f"  round {rounds}: sampling {args.chunk_size:,} "
            f"(pool {len(pool):,} so far)"
        )
        t0 = time.time()
        seqs = model.sample(
            n=args.chunk_size,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            pad_id=tok.pad_id,
            max_len=args.max_len,
            temperature=args.temperature,
            device=device,
            batch_size=args.batch,
        )
        total_raw += args.chunk_size
        decoded = [tok.decode(s) for s in seqs]
        decoded = [s for s in decoded if s]
        canon = canonicalize(decoded)
        valid = [c for c in canon if c is not None]
        total_valid += len(valid)
        added = 0
        for c in valid:
            if c in pool_set or c in train_set:
                continue
            pool_set.add(c)
            pool.append(c)
            added += 1
        print(
            f"    decoded={len(decoded)} valid={len(valid)} added={added} "
            f"pool={len(pool)}   ({time.time() - t0:.1f}s)"
        )

    print(f"final pool size: {len(pool):,} (raw sampled: {total_raw:,})")
    smiles = pool[: args.target_pool]  # cap

    # ChemNet activations on the pool
    print("computing ChemNet activations on pool ...")
    import fcd  # cpu

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    chemnet = fcd.load_ref_model()
    t0 = time.time()
    A = fcd.get_predictions(chemnet, smiles)
    print(f"  activations shape = {A.shape}   ({time.time() - t0:.1f}s)")

    with open(pool_path, "wb") as f:
        pickle.dump({"smiles": smiles, "activations": A}, f)
    print(f"wrote {pool_path} ({pool_path.stat().st_size / 1e6:.1f} MB)")

    summary = {
        "ckpt": str(args.ckpt),
        "target_pool": args.target_pool,
        "actual_pool": len(smiles),
        "rounds": rounds,
        "total_raw": total_raw,
        "total_valid_after_canon": total_valid,
        "temperature": args.temperature,
        "activation_shape": list(A.shape),
    }
    with open(art_dir / "build_pool_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
