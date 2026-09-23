import time
import torch
import tiktoken
from data_loader import create_dataloader

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

total_tokens = len(tokenizer.encode(raw_text))
print(f"Quantidade total de tokens no texto: {total_tokens}\n")


def rodar_experimento(nome, context_length, batch_size, embedding_dim):
    print(f"=== {nome} ===")
    print(f"context_length={context_length} | batch_size={batch_size} | embedding_dim={embedding_dim}")

    inicio = time.time()

    dataloader = create_dataloader(
        raw_text,
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=False
    )

    # quantidade de amostras geradas pelo Dataset
    quantidade_amostras = len(dataloader.dataset)

    inputs, targets = next(iter(dataloader))

    token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + pos_embeddings

    duracao = time.time() - inicio

    print(f"Quantidade de amostras (sequências) geradas: {quantidade_amostras}")
    print(f"Formato do lote de entrada:      {tuple(inputs.shape)}")
    print(f"Formato da entrada final do modelo: {tuple(input_embeddings.shape)}")
    print(f"Tempo de execução: {duracao:.4f}s")
    print()


# Variando o tamanho do contexto (context_length)
rodar_experimento("Contexto pequeno", context_length=4, batch_size=8, embedding_dim=256)
rodar_experimento("Contexto médio", context_length=16, batch_size=8, embedding_dim=256)
rodar_experimento("Contexto grande", context_length=64, batch_size=8, embedding_dim=256)

# Variando o tamanho do lote (batch_size)
rodar_experimento("Lote pequeno", context_length=16, batch_size=2, embedding_dim=256)
rodar_experimento("Lote grande", context_length=16, batch_size=32, embedding_dim=256)

# Variando a dimensão do embedding
rodar_experimento("Embedding pequeno", context_length=16, batch_size=8, embedding_dim=16)
rodar_experimento("Embedding grande", context_length=16, batch_size=8, embedding_dim=768)
