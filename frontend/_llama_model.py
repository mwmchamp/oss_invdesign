"""Minimal Llama 3 transformer for inference with KV-cache.

Loads Meta-format checkpoints (consolidated.00.pth) directly.
No HuggingFace dependency — just PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 3072
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: int = 8
    vocab_size: int = 128256
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 2048


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = args.n_heads // args.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV cache (populated during generation)
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis, use_cache: bool = False):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if use_cache:
            if self.cache_k is not None:
                xk = torch.cat([self.cache_k, xk], dim=1)
                xv = torch.cat([self.cache_v, xv], dim=1)
            self.cache_k = xk
            self.cache_v = xv

        # GQA: repeat k/v heads
        kv_seqlen = xk.shape[1]
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(bsz, kv_seqlen, self.n_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(bsz, kv_seqlen, self.n_heads, self.head_dim)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_heads, kv_seqlen, head_dim)
        xv = xv.transpose(1, 2)

        # For cached single-token generation, is_causal must be False
        is_causal = (seqlen > 1) and not use_cache
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden = int(2 * args.dim * 4 / 3)
        if args.ffn_dim_multiplier is not None:
            hidden = int(hidden * args.ffn_dim_multiplier)
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, freqs_cis, use_cache: bool = False):
        x = x + self.attention(self.attention_norm(x), freqs_cis, use_cache=use_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len, args.rope_theta)

    def reset_cache(self):
        for layer in self.layers:
            layer.attention.reset_cache()

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        """Forward pass with optional KV-cache.

        Args:
            tokens: (bsz, seqlen) input token ids
            start_pos: position offset for freqs_cis (used with KV-cache)
            use_cache: if True, use and update KV-cache for efficient generation
        Returns:
            logits: (bsz, seqlen, vocab_size)
        """
        seqlen = tokens.shape[1]
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen].to(h.device)
        for layer in self.layers:
            h = layer(h, freqs_cis, use_cache=use_cache)
        h = self.norm(h)
        return self.output(h)
