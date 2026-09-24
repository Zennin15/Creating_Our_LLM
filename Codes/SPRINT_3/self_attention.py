"""
Self-Attention com pesos treináveis (Wq, Wk, Wv).

Cada token de entrada é projetado em três vetores:
  query (Q) -> "o que este token procura?"
  key   (K) -> "o que este token oferece para ser encontrado?"
  value (V) -> "a informação que este token entrega se for atendido"

attention scores  = Q @ K^T
attention weights = softmax(scores / sqrt(d_k))      <- Scaled Dot-Product
context vector    = weights @ V

Aqui NÃO há máscara: cada token enxerga a sequência inteira (passado e futuro).
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False, scale=True):
        super().__init__()
        self.scale = scale  # scale=False desliga a divisão por sqrt(d_k) (usado nos experimentos)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x, return_weights=False):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        if self.scale:
            attn_scores = attn_scores / keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context_vec = attn_weights @ values
        if return_weights:
            return context_vec, attn_weights
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    # Mesmo exemplo do context_vector.py: "Your journey starts with one step"
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    batch = inputs.unsqueeze(0)  # (1, 6, 3)

    sa = SelfAttention(d_in=3, d_out=2)
    context, weights = sa(batch, return_weights=True)
    print("Formato do context vector:", context.shape)
    print("Attention weights:\n", weights[0])
    print("Cada linha soma 1:", weights[0].sum(dim=-1))
