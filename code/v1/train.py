"""v1: Train an LSTM SMILES language model on the (non-heldout) training corpus."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.paths import ARTIFACTS_DIR, version_dirs  # noqa: E402
from common.tokenizer import SmilesTokenizer  # noqa: E402
from model import SmilesLSTM  # noqa: E402

VERSION = "v1"


class TokenDataset(Dataset):
    """Holds pre-tokenized SMILES as a list of np.int16 1-D arrays."""

    def __init__(self, sequences: list[np.ndarray]):
        self.seqs = sequences

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.seqs[idx]


def make_collate(pad_id: int):
    def collate(batch: list[np.ndarray]):
        lens = [len(s) for s in batch]
        max_len = max(lens)
        out = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        for i, s in enumerate(batch):
            out[i, : len(s)] = s
        out_t = torch.from_numpy(out)
        # inputs are seq[:-1], targets are seq[1:]
        return out_t[:, :-1].contiguous(), out_t[:, 1:].contiguous()

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0, help="0 = use all SMILES")
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(ARTIFACTS_DIR / "v0" / "train_smiles.txt"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, art_dir, _ = version_dirs(VERSION)
    log_path = art_dir / "run.log"
    log_f = open(log_path, "w", buffering=1, encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_f.write(msg + "\n")

    log(f"args = {vars(args)}")

    # ---- load + tokenize training SMILES ----
    log(f"Reading {args.train_file} ...")
    with open(args.train_file) as f:
        smiles = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        smiles = smiles[: args.limit]
    log(f"  {len(smiles):,} SMILES")

    log("Building tokenizer from training SMILES ...")
    t0 = time.time()
    tok = SmilesTokenizer.build_from_smiles(smiles, min_freq=1)
    log(f"  vocab_size = {tok.vocab_size}  ({time.time() - t0:.1f}s)")
    tok.save(art_dir / "tokenizer.json")
    log(f"  saved tokenizer to {art_dir / 'tokenizer.json'}")

    log("Encoding sequences ...")
    t0 = time.time()
    n_too_long = 0
    sequences: list[np.ndarray] = []
    for s in smiles:
        ids = tok.encode(s)
        if len(ids) > args.max_len:
            n_too_long += 1
            ids = ids[: args.max_len - 1] + [tok.eos_id]
        sequences.append(np.asarray(ids, dtype=np.int16))
    log(
        f"  encoded {len(sequences):,} in {time.time() - t0:.1f}s "
        f"(truncated: {n_too_long})"
    )

    # ---- dataset / loader ----
    ds = TokenDataset(sequences)
    collate = make_collate(tok.pad_id)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows multiprocessing pain; pre-tokenized, so cheap anyway
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    log(f"  batches/epoch = {len(loader)}")

    # ---- model / opt ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device = {device}")
    model = SmilesLSTM(
        vocab_size=tok.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=tok.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ---- train ----
    best_loss = float("inf")
    log_interval = max(1, len(loader) // 20)
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        running_loss = 0.0
        running_tokens = 0
        for step, (inp, tgt) in enumerate(loader, start=1):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits, _ = model(inp)
                loss = loss_fn(
                    logits.reshape(-1, tok.vocab_size), tgt.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ntok = (tgt != tok.pad_id).sum().item()
            running_loss += loss.item() * ntok
            running_tokens += ntok
            if step % log_interval == 0 or step == len(loader):
                avg = running_loss / max(running_tokens, 1)
                elapsed = time.time() - ep_t0
                rate = step / elapsed
                log(
                    f"  epoch {epoch} step {step}/{len(loader)} "
                    f"loss={avg:.4f} ppl={np.exp(avg):.2f} "
                    f"rate={rate:.1f} bat/s elapsed={elapsed:.0f}s"
                )
        ep_loss = running_loss / max(running_tokens, 1)
        log(
            f"==== epoch {epoch} done in {time.time() - ep_t0:.1f}s  "
            f"avg_loss={ep_loss:.4f} ppl={np.exp(ep_loss):.2f} ===="
        )
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "tokenizer": tok.token_to_id,
            "epoch": epoch,
            "loss": ep_loss,
        }
        torch.save(ckpt, art_dir / f"ckpt_epoch{epoch}.pt")
        log(f"  wrote {art_dir / f'ckpt_epoch{epoch}.pt'}")
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(ckpt, art_dir / "ckpt_best.pt")
            log(f"  new best loss; updated ckpt_best.pt")

    summary = {
        "n_params": n_params,
        "vocab_size": tok.vocab_size,
        "best_loss": best_loss,
        "epochs_trained": args.epochs,
    }
    with open(art_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"summary = {summary}")
    log_f.close()


if __name__ == "__main__":
    main()
