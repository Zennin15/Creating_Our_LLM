"""
FeedForward Network — Capítulo 4, seção 4.3.

Pequena rede de 2 camadas lineares aplicada INDEPENDENTEMENTE a cada token:

    emb_dim  --Linear-->  ff_dim (=4 * emb_dim)  --GELU-->  --Linear-->  emb_dim

A atenção mistura informação ENTRE tokens; a FeedForward processa cada token
sozinho, com mais capacidade (a expansão 4x dá um espaço maior para a rede
representar combinações das features). É nela que fica ~2/3 dos parâmetros de
cada Transformer Block.

Entrada e saída: (batch, T, emb_dim) -> (batch, T, emb_dim)
O tamanho intermediário (ff_dim) pode ser mudado via cfg["ff_dim"] (usado nos
experimentos); se não existir no cfg, usa 4 * emb_dim como o GPT-2.
"""

import torch
import torch.nn as nn

from gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        ff_dim = cfg.get("ff_dim", 4 * emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),   # expande
            GELU(),                       # não linearidade
            nn.Linear(ff_dim, emb_dim),   # comprime de volta
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(123)
    cfg = {"emb_dim": 768}
    ff = FeedForward(cfg)
    x = torch.randn(2, 3, 768)  # (batch, T, emb_dim)
    print("Entrada:", x.shape, "-> Saída:", ff(x).shape)
    print("Camada interna: 768 -> 3072 -> 768")
