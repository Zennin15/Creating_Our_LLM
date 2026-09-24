"""
Testes de sanidade das implementações de atenção.
Verificam propriedades matemáticas e comparam com a implementação de referência do PyTorch.
"""

import torch
import torch.nn.functional as F

from self_attention import SelfAttention
from causal_attention import CausalAttention
from multi_head_attention import MultiHeadAttention

torch.manual_seed(123)
B, T, D_IN, D_OUT, HEADS = 2, 8, 16, 32, 4
x = torch.randn(B, T, D_IN)


def teste_formatos():
    assert SelfAttention(D_IN, D_OUT)(x).shape == (B, T, D_OUT)
    assert CausalAttention(D_IN, D_OUT, T)(x).shape == (B, T, D_OUT)
    out, w = MultiHeadAttention(D_IN, D_OUT, T, 0.0, HEADS)(x, return_weights=True)
    assert out.shape == (B, T, D_OUT) and w.shape == (B, HEADS, T, T)


def teste_linhas_somam_1():
    _, w = SelfAttention(D_IN, D_OUT)(x, return_weights=True)
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-6)
    _, w = MultiHeadAttention(D_IN, D_OUT, T, 0.0, HEADS)(x, return_weights=True)
    assert torch.allclose(w.sum(-1), torch.ones(B, HEADS, T), atol=1e-6)


def teste_mascara_causal_zera_futuro():
    _, w = CausalAttention(D_IN, D_OUT, T)(x, return_weights=True)
    assert torch.triu(w, diagonal=1).abs().sum() == 0
    assert torch.allclose(w[:, 0, 0], torch.ones(B))  # 1º token só vê a si mesmo
    _, w = MultiHeadAttention(D_IN, D_OUT, T, 0.0, HEADS)(x, return_weights=True)
    assert torch.triu(w, diagonal=1).abs().sum() == 0


def teste_sem_vazamento_do_futuro():
    """Alterar o token t não pode mudar a saída das posições < t (causal)."""
    for modulo in (CausalAttention(D_IN, D_OUT, T), MultiHeadAttention(D_IN, D_OUT, T, 0.0, HEADS)):
        x2 = x.clone()
        x2[:, 5:, :] += 10.0  # muda só o "futuro" a partir da posição 5!!
        assert torch.allclose(modulo(x)[:, :5], modulo(x2)[:, :5], atol=1e-5)


def teste_contra_referencia_pytorch():
    """Causal Attention == F.scaled_dot_product_attention(is_causal=True)."""
    ca = CausalAttention(D_IN, D_OUT, T).eval()
    q, k, v = ca.W_query(x), ca.W_key(x), ca.W_value(x)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ca(x), ref, atol=1e-5)

    """Multi-Head == referência do PyTorch por head + out_proj."""
    mha = MultiHeadAttention(D_IN, D_OUT, T, 0.0, HEADS).eval()
    hd = D_OUT // HEADS
    split = lambda t: t.view(B, T, HEADS, hd).transpose(1, 2)
    ref = F.scaled_dot_product_attention(split(mha.W_query(x)), split(mha.W_key(x)),
                                         split(mha.W_value(x)), is_causal=True)
    ref = mha.out_proj(ref.transpose(1, 2).reshape(B, T, D_OUT))
    assert torch.allclose(mha(x), ref, atol=1e-5)


def teste_heads_nao_mudam_parametros():
    n = lambda h: sum(p.numel() for p in MultiHeadAttention(D_IN, D_OUT, T, 0.0, h).parameters())
    assert n(1) == n(2) == n(4) == n(8)  # só muda como d_out é dividido


def teste_dropout_so_no_treino():
    mha = MultiHeadAttention(D_IN, D_OUT, T, 0.5, HEADS)
    mha.eval()
    assert torch.allclose(mha(x), mha(x))          # determinístico
    mha.train()
    assert not torch.allclose(mha(x), mha(x))      # dropout ativo


def teste_contexto_menor_que_context_length():
    ca = CausalAttention(D_IN, D_OUT, context_length=16)
    assert ca(x).shape == (B, T, D_OUT)            # T=8 < 16: máscara é recortada


if __name__ == "__main__":
    testes = [v for k, v in list(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(testes)} testes passaram.")
