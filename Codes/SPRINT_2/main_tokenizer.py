import urllib.request # Importa uma url
import re # Regular expression 
from tokenizer_v2 import SimpleTokenizerV2 # classe do tokenizador_v1

url = ("https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
 "the-verdict.txt")

file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# Tokenizer básico
# text = "Criando Nossa LLM--;.?_."
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text) # Separa o texto em tokens
# result = [item for item in result if item.strip()] # Remove os espaços
# print(result)

# 1. Tokenizer básico com o texto importado como entrada
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) # Separa o texto em tokens
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
# print(len(preprocessed)) # Retorna a quantidade de tokens no texto (sem espaços)
# print(preprocessed[:30]) # Retorna os 30 primeiros tokens.

# 2. Token iD
all_tokens = sorted(list(set(preprocessed))) # Constrói uma coleção de tokens únicos e ordena em ordem alfabética (adição: tranforma a coleção em uma lista)
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # Transformando a coleção em uma lista, conseguimos usar "extend" que adiciona elementos ao final da lista.
                                                # Nesse caso, adicionamos dois tokens: "<|endoftext|>" que indica o fim de um texto
                                                # e "<|unk|>", que indica um token que não está no vocabulário.

vocab = {token:integer for integer,token in enumerate(all_tokens)} # Constrói um vocabulário enumerado.

# for i, item in enumerate(vocab.items()):
#     print(item)

#    if i >= 50:
#        break

# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"
# Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids) # Exibe os token IDs
# print(tokenizer.decode(ids)) # Exibe o token a partir do ID

# Dois textos diferentes serão separados com endoftext
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# Palavras que não estão no vocabulário serão marcadas como unk
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))