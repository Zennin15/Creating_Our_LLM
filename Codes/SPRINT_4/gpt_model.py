"""
Modelo GPT simplificado — Capítulo 4, seções 4.4 e 4.6.

Pipeline completo (formatos entre parênteses; B = batch, T = nº de tokens):

  Token IDs                              (B, T)
    │  tok_emb  (Sprint 2)
    ▼
  Token embeddings                       (B, T, emb_dim)
    + pos_emb   (Sprint 2)  ← posições 0..T-1
    │  Dropout (drop_emb)
    ▼
  N x TransformerBlock                   (B, T, emb_dim)   ← Sprint 3 + Sprint 4
    │
    ▼
  LayerNorm final                        (B, T, emb_dim)
    │  out_head (Linear, sem bias)  = Language Model Head
    ▼
  Logits                                 (B, T, vocab_size)

Os LOGITS são uma pontuação bruta (sem normalizar) para cada token do
vocabulário, em CADA posição. Para prever o próximo token usamos os logits da
ÚLTIMA posição (ver generate.py).

Sobre o nº de parâmetros: aqui o out_head tem pesos próprios. O GPT-2 original
"amarra" (weight tying) os pesos do out_head aos da tok_emb, o que reduz de
~163M para ~124M. Não fazemos o tying para manter o código simples.
"""

import torch
import torch.nn as nn

from layer_norm import LayerNorm
from transformer_block import TransformerBlock

# Configuração do GPT-2 "small" (124M)
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # vocabulário do tokenizer BPE do GPT-2
    "context_length": 1024,  # nº máximo de tokens que o modelo enxerga
    "emb_dim": 768,          # dimensão dos embeddings
    "n_heads": 12,           # nº de heads da atenção
    "n_layers": 12,          # nº de Transformer Blocks empilhados
    "drop_rate": 0.1,        # probabilidade de dropout
    "qkv_bias": False,       # GPT-2 moderno não usa bias em Q, K, V
}


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Language Model Head: projeta emb_dim -> vocab_size (um logit por token)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):                         # in_idx: (B, T) Token IDs
        batch_size, seq_len = in_idx.shape
        assert seq_len <= self.context_length, (
            f"Sequência com {seq_len} tokens excede context_length={self.context_length}"
        )
        tok_embeds = self.tok_emb(in_idx)                                  # (B, T, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, emb_dim)
        x = tok_embeds + pos_embeds                                        # (B, T, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)                                             # (B, T, emb_dim)
        x = self.final_norm(x)                                             # (B, T, emb_dim)
        logits = self.out_head(x)                                          # (B, T, vocab_size)
        return logits


def contar_parametros(model):
    """Retorna um dicionário com o nº de parâmetros por componente."""
    n = lambda m: sum(p.numel() for p in m.parameters())
    blocos = model.trf_blocks
    return {
        "total": n(model),
        "tok_emb": n(model.tok_emb),
        "pos_emb": n(model.pos_emb),
        "blocos": n(blocos),
        "atencao": sum(n(b.att) for b in blocos),
        "feedforward": sum(n(b.ff) for b in blocos),
        "layernorm": sum(n(b.norm1) + n(b.norm2) for b in blocos) + n(model.final_norm),
        "out_head": n(model.out_head),
    }


if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # Duas "frases" de 4 tokens (IDs do tokenizer GPT-2)
    batch = torch.tensor([[6109, 3626, 6100, 345],
                          [6109, 1110, 6622, 257]])
    logits = model(batch)
    print("Entrada (Token IDs):", batch.shape)
    print("Saída (logits)     :", logits.shape, " -> (batch, T, vocab_size)")

    p = contar_parametros(model)
    print(f"\nParâmetros totais: {p['total']:,}")
    for k, v in p.items():
        if k != "total":
            print(f"  {k:<12} {v:>12,}  ({100 * v / p['total']:.1f}%)")
    print(f"\nCom weight tying (como o GPT-2): {p['total'] - p['out_head']:,}")
    print(f"Tamanho dos pesos (float32): {p['total'] * 4 / 1024**2:.1f} MB")
