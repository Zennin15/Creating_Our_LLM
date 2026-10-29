"""
Layer Normalization — Capítulo 4, seção 4.2.

Normaliza o vetor de CADA token (última dimensão, emb_dim) para ter média 0 e
variância 1, e depois aplica dois parâmetros treináveis (scale e shift) para
que o modelo possa "desfazer" a normalização se for útil.

Por que existe? Sem normalização, a escala das ativações pode crescer ou
encolher a cada camada, o que torna o treino instável (gradientes explodindo
ou sumindo). Com 12 ou mais blocos empilhados isso é crítico.

Diferença para BatchNorm: a LayerNorm calcula média/variância por token (ao
longo de emb_dim), então não depende do tamanho do batch nem de outros
exemplos — por isso é a escolha dos Transformers.

Entrada e saída: (batch, T, emb_dim) -> (batch, T, emb_dim)
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps  # evita divisão por zero quando a variância é ~0
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma (treinável)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta  (treinável)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False -> divide por N (e não N-1), igual ao nn.LayerNorm do PyTorch
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 5, 6)  # (batch, T, emb_dim)

    ln = LayerNorm(emb_dim=6)
    out = ln(x)

    print("Formato da saída:", out.shape)
    print("Média por token (≈ 0):\n", out.mean(dim=-1))
    print("Variância por token (≈ 1):\n", out.var(dim=-1, unbiased=False))
