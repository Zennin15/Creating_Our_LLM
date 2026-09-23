"""
Um Token ID é transformado em um vetor (representação numérica que a
rede consegue analisar, processar e "aprender").
"""

import torch

torch.manual_seed(123)

# Vocabulario com apenas 6 tokens (tokenizer do GPT-2: 50257 tokens)
vocab_size = 6

# Dimensão do vetor que vai representar cada token.
embedding_dim = 3


embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

print("Tabela de embeddings (pesos iniciais, aleatórios):")
print(embedding_layer.weight)

input_ids = torch.tensor([2, 3, 5, 1])

# Convertendo TokenID's para vetores
token_embeddings = embedding_layer(input_ids)

print("\nToken IDs de entrada:", input_ids)
print("\nVetores correspondentes (embeddings):")
print(token_embeddings)

print("\nFormato (shape) do resultado:", token_embeddings.shape)
# Cada TokenID virou um vetor do tamanho embedding_dim.
