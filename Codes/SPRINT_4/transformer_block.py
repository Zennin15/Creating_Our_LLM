"""
Transformer Block — Capítulo 4, seção 4.5.

Junta os componentes das Sprints anteriores com os novos:

    x ──► LayerNorm ──► Multi-Head Attention ──► Dropout ──► (+) ──► x'
    │                                                         ▲
    └─────────────────── conexão residual ───────────────────┘

    x' ─► LayerNorm ──► FeedForward ───────────► Dropout ──► (+) ──► saída
    │                                                         ▲
    └─────────────────── conexão residual ───────────────────┘

* Pre-LayerNorm: a normalização vem ANTES de cada sub-camada (como no GPT-2).
* Conexão residual: somamos a entrada da sub-camada à sua saída. Cria um
  "atalho" para o gradiente atravessar muitos blocos sem sumir, e faz cada
  sub-camada aprender só um "ajuste" sobre o que já existe.
* A Multi-Head Attention é a da Sprint 3 (Codes/SPRINT_3/multi_head_attention.py).

O bloco NÃO muda o formato do tensor:  (batch, T, emb_dim) -> (batch, T, emb_dim)
É isso que permite empilhar quantos blocos quisermos.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Reaproveita a Multi-Head Attention da Sprint 3
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SPRINT_3"))
from multi_head_attention import MultiHeadAttention  # noqa: E402

from feed_forward import FeedForward  # noqa: E402
from layer_norm import LayerNorm  # noqa: E402


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):                      # x: (batch, T, emb_dim)
        # --- Sub-bloco 1: atenção ---
        shortcut = x
        x = self.norm1(x)                      # (batch, T, emb_dim)
        x = self.att(x)                        # (batch, T, emb_dim)
        x = self.drop_shortcut(x)
        x = x + shortcut                       # residual

        # --- Sub-bloco 2: FeedForward ---
        shortcut = x
        x = self.norm2(x)                      # (batch, T, emb_dim)
        x = self.ff(x)                         # (batch, T, emb_dim)
        x = self.drop_shortcut(x)
        x = x + shortcut                       # residual
        return x


if __name__ == "__main__":
    torch.manual_seed(123)
    cfg = {"emb_dim": 768, "context_length": 1024, "n_heads": 12,
           "drop_rate": 0.1, "qkv_bias": False}
    block = TransformerBlock(cfg)
    x = torch.rand(2, 4, 768)  # (batch, T, emb_dim)
    out = block(x)
    print("Entrada:", x.shape)
    print("Saída  :", out.shape, "(o bloco preserva o formato)")
    n = sum(p.numel() for p in block.parameters())
    n_att = sum(p.numel() for p in block.att.parameters())
    n_ff = sum(p.numel() for p in block.ff.parameters())
    print(f"Parâmetros do bloco: {n:,} (atenção: {n_att:,} | FeedForward: {n_ff:,})")
