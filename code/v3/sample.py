"""v3: Sample from the trained scaled Transformer LM, post-process, verify, FCD."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.local_fcd import local_fcd  # noqa: E402
from common.paths import ARTIFACTS_DIR, EVAL_TRAIN_FILE, version_dirs  # noqa: E402
from common.postprocess import canonicalize, load_train_set, write_submission  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402
from common.verify import assert_thresholds, official_aux_metrics  # noqa: E402
from model import TransformerLM  # noqa: E402

VERSION = "v3"


def adaptive_sample(
    model,
    tok: SmilesTokenizer,
    train_set: set[str],
    target_n: int,
    chunk_size: int,
    temperature: float,
    max_len: int,
    batch: int,
    device: torch.device,
    max_rounds: int = 10,
) -> tuple[list[str], dict]:
    pool_set: set[str] = set()
    pool: list[str] = []
    total_raw = 0
    total_decoded = 0
    total_valid = 0
    total_canon_drop = 0
    rounds = 0

    while len(pool) < target_n and rounds < max_rounds:
        rounds += 1
        print(
            f"  round {rounds}: sampling {chunk_size:,} (pool has {len(pool):,} so far)"
        )
        t0 = time.time()
        seqs = model.sample(
            n=chunk_size,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            pad_id=tok.pad_id,
            max_len=max_len,
            temperature=temperature,
            device=device,
            batch_size=batch,
        )
        total_raw += chunk_size
        decoded = [tok.decode(s) for s in seqs]
        decoded = [s for s in decoded if s]
        total_decoded += len(decoded)
        canon = canonicalize(decoded)
        valid = [c for c in canon if c is not None]
        total_valid += len(valid)
        total_canon_drop += len(canon) - len(valid)
        added = 0
        for c in valid:
            if c in pool_set:
                continue
            if c in train_set:
                continue
            pool_set.add(c)
            pool.append(c)
            added += 1
        print(
            f"    decoded={len(decoded)}  valid={len(valid)}  "
            f"new_pool_adds={added}  pool={len(pool)}   "
            f"({time.time() - t0:.1f}s)"
        )

    if len(pool) < target_n:
        raise RuntimeError(
            f"After {rounds} rounds and {total_raw:,} raw samples, only "
            f"{len(pool):,} valid+unique+novel SMILES — need {target_n}."
        )

    sub = pool[:target_n]
    stats = {
        "rounds": rounds,
        "total_raw": total_raw,
        "total_decoded": total_decoded,
        "total_valid_after_canon": total_valid,
        "total_canon_invalid": total_canon_drop,
        "pool_size_final": len(pool),
        "n_submitted": len(sub),
        "validity_rate_overall": total_valid / max(total_raw, 1),
    }
    return sub, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt_best.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=15_000)
    parser.add_argument("--target_n", type=int, default=10_000)
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, art_dir, pred_dir = version_dirs(VERSION)
    ckpt_path = art_dir / args.ckpt
    if not ckpt_path.exists():
        print(f"missing checkpoint {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    if args.seed:
        torch.manual_seed(args.seed)

    print(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    print("loading eval-bundle train set ...")
    train_set = load_train_set(EVAL_TRAIN_FILE)
    print(f"  {len(train_set):,} train SMILES")

    print(
        f"adaptive sampling (T={args.temperature}, target_n={args.target_n}, "
        f"chunk={args.chunk_size}, max_len={args.max_len}, batch={args.batch})"
    )
    sub, sample_stats = adaptive_sample(
        model=model,
        tok=tok,
        train_set=train_set,
        target_n=args.target_n,
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        max_len=args.max_len,
        batch=args.batch,
        device=device,
    )
    print(f"  sample stats = {json.dumps(sample_stats, indent=2)}")

    tag = args.tag or f"T{args.temperature}"
    out_path = pred_dir / f"v3_{tag}.txt"
    write_submission(sub, out_path)
    print(f"  wrote {out_path}")

    print("running official aux-metric verification on submission file ...")
    metrics = official_aux_metrics(out_path)
    for k, v in metrics.items():
        print(f"  {k:>12} = {v}")
    assert_thresholds(metrics, threshold=args.threshold)
    print(f"  aux thresholds (>= {args.threshold}) PASSED")

    print("computing local FCD ...")
    t0 = time.time()
    fcd_val, _ = local_fcd(sub, ARTIFACTS_DIR / "local_fcd_ref.pkl")
    print(f"  local FCD = {fcd_val:.4f}   ({time.time() - t0:.1f}s)")

    summary = {
        "checkpoint": str(ckpt_path.name),
        "temperature": args.temperature,
        "target_n": args.target_n,
        "chunk_size": args.chunk_size,
        "sample_stats": sample_stats,
        "official_aux_metrics": metrics,
        "aux_threshold": args.threshold,
        "local_fcd": fcd_val,
        "submission_file": str(out_path.relative_to(out_path.parents[3])),
    }
    sum_path = pred_dir / f"v3_{tag}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {sum_path}")


if __name__ == "__main__":
    main()
