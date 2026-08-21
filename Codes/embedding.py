import urllib.request # Importa uma url
import re # Regular expression 

url = ("https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
 "the-verdict.txt")

file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Tokenizer básico
# text = "Criando Nossa LLM--;.?_."
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text) # Separa o texto em tokens
# result = [item for item in result if item.strip()] # Remove os espaços
# print(result)

# 1. Tokenizer básico com o texto importado como entrada
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) # Separa o texto em tokens
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
print(len(preprocessed)) # Retorna a quantidade de tokens no texto (sem espaços)
print(preprocessed[:30]) # Retorna os 30 primeiros tokens.

# 2. Token iD
all_words = sorted(set(preprocessed)) # Constrói uma coleção de tokens únicos e ordena em ordem alfabética
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)} # Constrói um vocabulário enumerado.

for i, item in enumerate(vocab.items()):
    print(item)

    if i >= 50:
        break