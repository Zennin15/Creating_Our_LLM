"""
Testes de sanidade da Sprint 4 (LayerNorm, GELU, FeedForward, TransformerBlock,
GPTModel e geração). Não precisam de internet nem de tokenizer: usam Token IDs
aleatórios e um modelo pequeno.

Como rodar (a partir da RAIZ do repositório):
    python Codes/SPRINT_4/test_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_norm import LayerNorm
from gelu import GELU
from feed_forward import FeedForward
from transformer_block import TransformerBlock
from gpt_model import GPTModel, GPT_CONFIG_124M, contar_parametros
from generate import generate, generate_text_simple

torch.manual_seed(123)
CFG = {"vocab_size": 100, "context_length": 16, "emb_dim": 32, "n_heads": 4,
       "n_layers": 2, "drop_rate": 0.1, "qkv_bias": False}
B, T = 2, 8
ids = torch.randint(0, CFG["vocab_size"], (B, T))


def teste_layernorm_media0_var1():
    out = LayerNorm(CFG["emb_dim"])(torch.randn(B, T, CFG["emb_dim"]) * 5 + 3)
    assert torch.allclose(out.mean(-1), torch.zeros(B, T), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(B, T), atol=1e-3)


def teste_layernorm_igual_pytorch():
    x = torch.randn(B, T, CFG["emb_dim"])
    assert torch.allclose(LayerNorm(CFG["emb_dim"])(x), nn.LayerNorm(CFG["emb_dim"])(x), atol=1e-5)


def teste_gelu_igual_pytorch():
    x = torch.linspace(-5, 5, 101)
    assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def teste_feedforward_formato_e_expansao():
    ff = FeedForward(CFG)
    assert ff(torch.randn(B, T, CFG["emb_dim"])).shape == (B, T, CFG["emb_dim"])
    assert ff.layers[0].out_features == 4 * CFG["emb_dim"]  # expansão 4x
    assert FeedForward({**CFG, "ff_dim": 48}).layers[0].out_features == 48


def teste_bloco_preserva_formato():
    x = torch.randn(B, T, CFG["emb_dim"])
    assert TransformerBlock(CFG)(x).shape == x.shape


def teste_conexao_residual():
    """Se as sub-camadas produzem 0, o bloco tem que devolver a própria entrada."""
    bloco = TransformerBlock({**CFG, "drop_rate": 0.0}).eval()
    with torch.no_grad():
        bloco.att.out_proj.weight.zero_(); bloco.att.out_proj.bias.zero_()
        bloco.ff.layers[2].weight.zero_(); bloco.ff.layers[2].bias.zero_()
    x = torch.randn(B, T, CFG["emb_dim"])
    assert torch.allclose(bloco(x), x, atol=1e-6)


def teste_gpt_formato_dos_logits():
    assert GPTModel(CFG)(ids).shape == (B, T, CFG["vocab_size"])


def teste_gpt_sem_vazamento_do_futuro():
    """Mudar tokens a partir da posição 5 não pode alterar os logits das posições < 5."""
    m = GPTModel(CFG).eval()
    ids2 = ids.clone()
    ids2[:, 5:] = (ids2[:, 5:] + 1) % CFG["vocab_size"]
    with torch.no_grad():
        assert torch.allclose(m(ids)[:, :5], m(ids2)[:, :5], atol=1e-5)


def teste_gpt_rejeita_sequencia_longa():
    m = GPTModel(CFG)
    try:
        m(torch.randint(0, 100, (1, CFG["context_length"] + 1)))
    except AssertionError:
        return
    raise AssertionError("deveria ter recusado sequência maior que context_length")


def teste_dropout_so_no_treino():
    m = GPTModel({**CFG, "drop_rate": 0.5})
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(ids), m(ids))          # eval: determinístico
    m.train()
    with torch.no_grad():
        assert not torch.equal(m(ids), m(ids))      # train: dropout muda a saída


def teste_contagem_de_parametros_124M():
    """O livro reporta 163.009.536 parâmetros (sem weight tying) para o GPT-2 small."""
    p = contar_parametros(GPTModel(GPT_CONFIG_124M))
    assert p["total"] == 163_009_536, p["total"]
    assert p["total"] - p["out_head"] == 124_412_160   # com weight tying = "124M"
    assert p["total"] == (p["tok_emb"] + p["pos_emb"] + p["blocos"]
                          + p["out_head"] + 2 * GPT_CONFIG_124M["emb_dim"])


def teste_geracao_tamanho_e_prompt_preservado():
    m = GPTModel(CFG)
    out = generate_text_simple(m, ids, max_new_tokens=5, context_size=CFG["context_length"])
    assert out.shape == (B, T + 5)
    assert torch.equal(out[:, :T], ids)  # o prompt não é alterado


def teste_geracao_greedy_e_deterministica():
    m = GPTModel(CFG)
    a = generate(m, ids, 6, CFG["context_length"])
    b = generate(m, ids, 6, CFG["context_length"])
    assert torch.equal(a, b)


def teste_topk1_equivale_ao_greedy():
    m = GPTModel(CFG)
    greedy = generate(m, ids, 6, CFG["context_length"])
    torch.manual_seed(0)
    topk1 = generate(m, ids, 6, CFG["context_length"], temperature=1.5, top_k=1)
    assert torch.equal(greedy, topk1)


def teste_amostragem_reprodutivel_com_seed():
    m = GPTModel(CFG)
    torch.manual_seed(7); a = generate(m, ids, 6, CFG["context_length"], temperature=1.0)
    torch.manual_seed(7); b = generate(m, ids, 6, CFG["context_length"], temperature=1.0)
    assert torch.equal(a, b)


def teste_topk_so_sorteia_entre_os_k_melhores():
    m = GPTModel(CFG).eval()
    k = 3
    prompt = ids[:1]
    with torch.no_grad():
        permitidos = set(torch.topk(m(prompt)[0, -1], k).indices.tolist())
    for seed in range(30):
        torch.manual_seed(seed)
        prox = generate(m, prompt, 1, CFG["context_length"], temperature=5.0, top_k=k)[0, -1].item()
        assert prox in permitidos


def teste_geracao_alem_do_contexto():
    """Gerar mais tokens que o context_length: o modelo passa a usar só a janela final."""
    m = GPTModel(CFG)
    out = generate(m, ids, CFG["context_length"] + 10, CFG["context_length"])
    assert out.shape == (B, T + CFG["context_length"] + 10)


def teste_geracao_restaura_modo_treino():
    m = GPTModel(CFG); m.train()
    generate(m, ids, 2, CFG["context_length"])
    assert m.training


def teste_eos_interrompe():
    m = GPTModel(CFG)
    primeiro = generate(m, ids[:1], 1, CFG["context_length"])[0, -1].item()
    out = generate(m, ids[:1], 10, CFG["context_length"], eos_id=primeiro)
    assert out.shape[1] == ids.shape[1]  # parou antes de anexar o 1º token


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"OK  {nome}")
    print(f"\n{len(testes)} testes passaram.")
