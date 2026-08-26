# Nesta implementação foi usado o tiktoken, uma biblioteca que implementa um tokenizer eficiente, seguindo o que foi implementado manualmente em 
# "main_tokenizer" e "tokenizer_v2" (para fins didáticos).

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") # Tokenizer do tipo gpt2

with open("the-verdict.txt", "r", encoding="utf-8") as f: # Importa o texto usado para o treinamento
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text) # Token ID
# print(len(enc_text))

enc_sample = enc_text[50:]

# Implementação das entradas e saídas (ou alvo) que serão utilizados na predição das próximas palavras em uma sequência textual.

context_size = 4 # Quantidade de tokens usados na entrada.
x = enc_sample[:context_size] # Entrada (de 0 até 4)
y = enc_sample[1:context_size + 1] # Alvo (de 1 até 5). Uma posição para a direita.

print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "-->", desired)

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "-->", tokenizer.decode([desired]))