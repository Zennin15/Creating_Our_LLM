import torch

# Implementação do cálculo de context vector para o segundo token (x^2) da sequência de entrada para fins didáticos.
# Primeiro calculamos os attention scores para posteriormente calcular os attention weights normalizando cada linha para que
# o valor em cada linha some exatamente 1.

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# Implementação manual da normalização das attention scores
"""
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
"""

# Implementação manual do softmax_naive para a normalização
"""
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
"""

# Função nativa do PyTorch para o softmax (mais eficiente)
attn_weights_2 = torch.softmax(attn_scores_2, dim = 0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1]

context_vec_2 = torch.zeros(query.shape)

for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)

# Implementação do context vector para todos os tokens da sequência de entrada (generalização).

# Calculando attention score
attn_scores = inputs @ inputs.T # multiplicação de matrizes (obs: T é a matriz transposta)
print(attn_scores, "\n")

# Calcula attention weights
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights, "\n")

# Calcula o context vector
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)