"""
Causal Attention (atenção causal / mascarada).

Em um LLM tipo GPT, o token na posição i só pode "olhar" para os tokens
0..i (passado e ele mesmo). Se pudesse ver o futuro, prever o próximo token
seria errado (ele já estaria na entrada).

Como funciona:
  1. Calculamos os attention scores (Q @ K^T) como na Self-Attention.
  2. Os scores ACIMA da diagonal (futuro) são trocados por -inf.
  3. O softmax transforma -inf em peso 0 e renormaliza cada linha para somar 1.
  4. Dropout zera aleatoriamente alguns pesos durante o treino.
"""

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False, scale=True):
        super().__init__()
        self.d_out = d_out
        self.scale = scale  # scale=False desliga a divisão por sqrt(d_k) (usado nos experimentos)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Máscara triangular superior (sem a diagonal): 1 = futuro = bloqueado.
        # register_buffer -> a máscara acompanha o modelo (.to(device), state_dict)
        # sem ser tratada como parâmetro treinável.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, return_weights=False):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # (batch, T, T)
        # Recorta a máscara para o tamanho real da sequência (T <= context_length)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        if self.scale:
            attn_scores = attn_scores / keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        context_vec = attn_weights_drop @ values
        if return_weights:
            # devolve os pesos SEM dropout, para visualização
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
    batch = torch.stack((inputs, inputs), dim=0)  # 2 entradas iguais -> (2, 6, 3)

    context_length = batch.shape[1]
    ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)
    context_vecs, weights = ca(batch, return_weights=True)

    print("Formato do batch:", batch.shape)
    print("Formato dos context vectors:", context_vecs.shape)
    print("Attention weights (triângulo superior = 0):\n", weights[0])
    print("Cada linha soma 1:", weights[0].sum(dim=-1))
