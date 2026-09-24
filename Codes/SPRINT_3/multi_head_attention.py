"""
Multi-Head Attention — Capítulo 3, seção 3.6.

Em vez de uma única atenção, usamos várias "cabeças" (heads) em paralelo.
Cada head tem seus próprios Q, K, V e pode aprender um tipo diferente de
relação entre os tokens. Depois, as saídas das heads são concatenadas e
combinadas por uma projeção linear final (out_proj).

Implementação sem loop sobre as heads:
  1. Projeta a entrada com UMA matriz Wq/Wk/Wv de tamanho (d_in, d_out).
  2. Reorganiza o resultado de (b, T, d_out) para (b, num_heads, T, head_dim),
     onde head_dim = d_out // num_heads.
  3. Faz a atenção causal em todas as heads de uma vez (multiplicação em lote).
  4. Junta as heads de volta em (b, T, d_out) e aplica out_proj.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, scale=True):
        super().__init__()
        assert d_out % num_heads == 0, "d_out deve ser divisível por num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # dimensão de cada head
        self.scale = scale  # scale=False desliga a divisão por sqrt(head_dim) (usado para os experimentos)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # combina a saída das heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, return_weights=False):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)        
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, T, T)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        if self.scale:
            attn_scores = attn_scores / keys.shape[-1] ** 0.5  # keys.shape[-1] = head_dim
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # (b, num_heads, T, head_dim) -> (b, T, num_heads, head_dim)
        context_vec = (attn_weights_drop @ values).transpose(1, 2)
        # concatena as heads: (b, T, num_heads * head_dim) = (b, T, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_weights:
            # pesos SEM dropout, formato (b, num_heads, T, T)
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
    batch = torch.stack((inputs, inputs), dim=0)  # (2, 6, 3)

    context_length = batch.shape[1]
    mha = MultiHeadAttention(d_in=3, d_out=2, context_length=context_length,
                             dropout=0.0, num_heads=2)
    context_vecs, weights = mha(batch, return_weights=True)

    print("Formato dos context vectors:", context_vecs.shape)  
    print("Formato dos attention weights:", weights.shape)
    print("head_dim =", mha.head_dim)
