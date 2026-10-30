"""
GELU (Gaussian Error Linear Unit) — Capítulo 4, seção 4.3.

É a função de ativação usada na FeedForward do GPT-2. Parecida com a ReLU, mas
suave: em vez de cortar tudo que é negativo de uma vez (ReLU), ela deixa passar
uma fração pequena de valores negativos. Isso dá gradientes mais bem
comportados perto de zero.

Usamos a aproximação por tanh (a mesma do GPT-2):
    GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

Entrada e saída: mesmo formato (a função é aplicada elemento a elemento).
"""

import math
import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))
        )


if __name__ == "__main__":
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("x    :", x)
    print("GELU :", GELU()(x))
    print("ReLU :", torch.relu(x))
