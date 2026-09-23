import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    Recebe um texto bruto e um tokenizer, e produz pares entrada / alvo
    prontos para receber um treinamento: o alvo é sempre a entrada deslocada em 1 token
    para a direita prevendo o próximo token.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # max_length como entrada e o mesmo pedaço deslocado 1 posição como alvo.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128,
                          shuffle=True, drop_last=True, num_workers=0):
    """
    Recebe o texto bruto e devolve um DataLoader
    (do PyTorch) já pronto, que entrega os dados em lotes.
    batch_size -> quantas sequências vêm em cada lote
    max_length -> quantos tokens por sequência
    stride     -> de quanto em quanto a janela "anda" sobre o texto
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Entradas:\n", inputs)
    print("Alvos:\n", targets)
