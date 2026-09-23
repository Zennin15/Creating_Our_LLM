# Usado o tiktoken, uma biblioteca que implementa um tokenizer mais eficiente. 

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") # Tokenizer do tipo gpt2

with open("the-verdict.txt", "r", encoding="utf-8") as f: # Importa o texto usado 
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text) 

enc_sample = enc_text[50:]

# entradas e saídas que serão utilizados para "prever" as próximas palavras da sequência.

context_size = 4 # Qtde de tokens usados na entrada.
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1] 

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