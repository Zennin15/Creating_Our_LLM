"""
Positional Embeddings: a embedding de um token é sempre a mesma, não
importa a posição dele na frase. Pra resolver isso, somamos uma segunda
embedding que representa só a POSIÇÃO (0, 1, 2, 3...).
"""

import torch
from data_loader import create_dataloader

torch.manual_seed(123)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
embedding_dim = 256
context_length = 4
batch_size = 8

# Lote de dados (Token IDs)
dataloader = create_dataloader(
    raw_text, batch_size=batch_size, max_length=context_length,
    stride=context_length, shuffle=False
)
inputs, targets = next(iter(dataloader))
print("Formato do lote de entradas (Token IDs):", inputs.shape)

# Token embeddings
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding_layer(inputs)
print("Formato dos Token Embeddings:", token_embeddings.shape)

# Positional embeddings: uma entrada por posição dentro do contexto
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)
positions = torch.arange(context_length)
pos_embeddings = pos_embedding_layer(positions)
print("Formato dos Positional Embeddings:", pos_embeddings.shape)

# Soma (broadcast automático do PyTorch)
input_embeddings = token_embeddings + pos_embeddings
print("Formato final (entrada pronta para o modelo):", input_embeddings.shape)
