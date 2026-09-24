import torch
import tiktoken
from data_loader import create_dataloader

torch.manual_seed(123)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total de caracteres no texto: {len(raw_text)}")

# Tokenização -> Token IDs
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
print(f"Tamanho do vocabulário do tokenizer: {vocab_size}")

token_ids_exemplo = tokenizer.encode(raw_text[:50])
print(f"\nExemplo - trecho de texto: {raw_text[:50]!r}")
print(f"Token IDs correspondentes: {token_ids_exemplo}")

# Parâmetros do pipeline
context_length = 4
batch_size = 8
embedding_dim = 256
stride = context_length

# Sequências de treinamento + lote de dados
dataloader = create_dataloader(
    raw_text,
    batch_size=batch_size,
    max_length=context_length,
    stride=stride,
    shuffle=False
)

inputs, targets = next(iter(dataloader))

print("\n--- Um lote de dados ---")
print("Entradas (Token IDs):\n", inputs)
print("Alvos    (Token IDs):\n", targets)
print("Formato do lote de entradas:", inputs.shape)

# Embeddings
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding_layer(inputs)
print("\nFormato dos Token Embeddings:", token_embeddings.shape)

# Positional Embeddings
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Formato dos Positional Embeddings:", pos_embeddings.shape)


input_embeddings = token_embeddings + pos_embeddings
print("\nFormato final da entrada do modelo:", input_embeddings.shape)
