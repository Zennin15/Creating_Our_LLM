"""
Avaliação quantitativa do modelo — Capítulo 5, seções 5.1.2 e 5.1.3.

O que o modelo produz: LOGITS (B, T, vocab_size). Para cada posição, uma
pontuação bruta para cada token do vocabulário.
O que queremos medir: quão bem esses logits "apostam" no token que veio de fato
depois (o alvo). A medida usada é a CROSS-ENTROPY (entropia cruzada):

    1. softmax(logits)             -> probabilidades por token do vocabulário
    2. p_alvo = probabilidade que o modelo deu AO TOKEN CORRETO
    3. loss   = -log(p_alvo)       -> média sobre todas as posições e lotes

Intuição:
    p_alvo = 1.0   -> loss = 0        (certeza no token certo)
    p_alvo = 0.5   -> loss = 0.69
    p_alvo = 1/V   -> loss = log(V)   (modelo "chutando" no escuro, uniforme)
    p_alvo -> 0    -> loss -> infinito

Um modelo SEM treino (pesos aleatórios) dá probabilidades ~uniformes: a loss
inicial esperada é log(50257) ~ 10.82. Treinar = empurrar essa loss para baixo.

PERPLEXITY = exp(loss). Interpretação: "em média, o modelo está tão incerto
quanto se tivesse que escolher ao acaso entre `perplexity` tokens igualmente
prováveis". Modelo aleatório: perplexity ~ vocab_size (50257). Modelo perfeito: 1.
"""

import math

import torch
import torch.nn.functional as F


def cross_entropy_manual(logits, targets):
    """
    Cross-entropy "na mão", só para mostrar o que o PyTorch faz por baixo.
    logits: (N, vocab_size)   targets: (N,)  com os IDs corretos.
    Equivale a F.cross_entropy(logits, targets) (ver test_training.py).
    """
    probs = torch.softmax(logits, dim=-1)                                  # (N, V)
    p_alvo = probs[torch.arange(targets.shape[0]), targets]                # (N,)
    return -torch.log(p_alvo).mean()


def calc_loss_batch(input_batch, target_batch, model, device):
    """Loss (cross-entropy) de UM lote. Mantém o grafo: serve para o backward."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)                                  # (B, T, V)
    # cross_entropy espera (N, V) e (N,): juntamos batch e tempo (flatten)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Loss MÉDIA sobre um DataLoader (treino ou validação).
    num_batches=None -> usa todos os lotes; senão, só os primeiros `num_batches`
    (mais rápido; é uma estimativa).
    Não liga/desliga o modo eval e não desliga gradiente: quem chama decide
    (ver avaliar_modelo).
    """
    if len(data_loader) == 0:
        return float("nan")
    n = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total = 0.0
    for i, (x, y) in enumerate(data_loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n


def avaliar_modelo(model, train_loader, val_loader, device, eval_iter=None):
    """
    Loss de treino e de validação com o modelo em modo de avaliação
    (dropout desligado) e sem gradiente. Restaura o modo anterior no fim.
    """
    estava_treinando = model.training
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train(estava_treinando)
    return train_loss, val_loss


def perplexity(loss):
    """perplexity = exp(loss). Protege contra overflow quando a loss é enorme."""
    if loss != loss:            # NaN
        return float("nan")
    return math.exp(min(loss, 700.0))


def loss_inicial_esperada(vocab_size):
    """Loss de um modelo que atribui probabilidade uniforme a todos os tokens."""
    return math.log(vocab_size)


if __name__ == "__main__":
    # Demonstração com logits inventados (não precisa de modelo nem tokenizer).
    torch.manual_seed(123)
    V, N = 10, 6
    logits = torch.randn(N, V)
    alvos = torch.randint(0, V, (N,))
    print("cross-entropy manual :", round(cross_entropy_manual(logits, alvos).item(), 6))
    print("F.cross_entropy      :", round(F.cross_entropy(logits, alvos).item(), 6))
    print(f"\nLogits iguais (modelo indeciso) com V={V}: "
          f"loss = {F.cross_entropy(torch.zeros(N, V), alvos).item():.4f}"
          f"  (log V = {loss_inicial_esperada(V):.4f}),"
          f" perplexity = {perplexity(loss_inicial_esperada(V)):.2f}  (= V)")
