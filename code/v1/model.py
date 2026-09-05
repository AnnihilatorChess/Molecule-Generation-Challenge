"""v1: LSTM language model over regex-tokenized SMILES."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmilesLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        return self.head(out), hidden

    @torch.no_grad()
    def sample(
        self,
        n: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_len: int = 110,
        temperature: float = 1.0,
        device: torch.device | str = "cuda",
        batch_size: int = 1024,
    ) -> list[list[int]]:
        """Batched temperature sampling. Returns list of length n, each a list of token ids
        (excluding BOS, up to and excluding EOS)."""
        self.eval()
        sequences: list[list[int]] = []
        remaining = n
        while remaining > 0:
            bs = min(batch_size, remaining)
            tokens = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
            done = torch.zeros(bs, dtype=torch.bool, device=device)
            outs = [[] for _ in range(bs)]
            hidden = None
            inp = tokens
            for _ in range(max_len):
                logits, hidden = self.forward(inp, hidden)
                last = logits[:, -1, :] / temperature
                probs = F.softmax(last, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)  # (bs, 1)
                nxt_cpu = nxt.squeeze(1).tolist()
                done_cpu = done.tolist()
                for i, tok in enumerate(nxt_cpu):
                    if done_cpu[i]:
                        continue
                    if tok == eos_id:
                        done[i] = True
                    else:
                        outs[i].append(tok)
                if done.all():
                    break
                inp = nxt
            sequences.extend(outs)
            remaining -= bs
        return sequences
