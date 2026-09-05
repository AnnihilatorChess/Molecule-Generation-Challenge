"""v2: GPT-style causal Transformer LM over regex-tokenized SMILES.

Uses `F.scaled_dot_product_attention` (PyTorch 2.x flash-attn / mem-efficient
backend) for training and an explicit KV-cache for autoregressive sampling so
that generation cost is O(T) instead of O(T^2) per sequence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Training/full-seq: is_causal handles the mask.
        # KV-cache step: q has T_new tokens, k/v have full history; no mask needed
        # because q can legally attend to all of k (everything up to its own position).
        is_causal = kv_cache is None and T > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), new_cache


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, kv_cache=None):
        a, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 110,
        dim: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim = dim
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: list | None = None,
        position_offset: int = 0,
    ):
        B, T = x.shape
        positions = torch.arange(
            position_offset, position_offset + T, device=x.device
        )
        h = self.tok_emb(x) + self.pos_emb(positions)
        h = self.drop(h)
        new_caches: list = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(h, kv_cache=cache)
            new_caches.append(new_cache)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits, new_caches

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
        batch_size: int = 512,
    ) -> list[list[int]]:
        self.eval()
        sequences: list[list[int]] = []
        remaining = n
        while remaining > 0:
            bs = min(batch_size, remaining)
            tokens = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
            done = torch.zeros(bs, dtype=torch.bool, device=device)
            outs = [[] for _ in range(bs)]
            kv_caches = None
            for step in range(max_len):
                logits, kv_caches = self.forward(
                    tokens, kv_caches=kv_caches, position_offset=step
                )
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
                tokens = nxt  # feed only the new token next step
            sequences.extend(outs)
            remaining -= bs
        return sequences
