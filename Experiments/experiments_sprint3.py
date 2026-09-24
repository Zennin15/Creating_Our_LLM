"""
Experimentos da Sprint 3 — comportamento do mecanismo de Attention.

Como rodar (a partir da RAIZ do repositório, onde está o the-verdict.txt):
    python Experiments/experiments_sprint3.py

Entrada: os mesmos dados da Sprint 2 (the-verdict.txt -> tokenizer GPT-2 ->
create_dataloader -> token embeddings + positional embeddings).

Saídas (em Experiments/results/sprint3/):
    figuras/*.png         matrizes de atenção
    metricas.csv          todas as métricas dos experimentos
    resumo_metricas.txt   tabelas em texto simples, prontas para consultar na análise

IMPORTANTE: as camadas de atenção aqui NÃO foram treinadas (pesos aleatórios,
seed fixa). Os padrões mostram o comportamento estrutural do mecanismo
(escala, máscara, heads), não relações linguísticas aprendidas.

Métricas (calculadas por linha da matriz de atenção):
    entropia_norm   0 = atenção concentrada em 1 token, 1 = atenção uniforme
                    (entropia dividida por log do nº de tokens visíveis)
    peso_max        maior peso da linha (média das linhas)
    A 1ª linha da atenção causal é ignorada (só enxerga a si mesma: peso = 1).
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # salva em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "Codes" / "SPRINT_2"))
sys.path.insert(0, str(RAIZ / "Codes" / "SPRINT_3"))

from data_loader import create_dataloader  # noqa: E402  (Sprint 2)
from self_attention import SelfAttention  # noqa: E402
from causal_attention import CausalAttention  # noqa: E402
from multi_head_attention import MultiHeadAttention  # noqa: E402
from attention_viz import (plot_attention_matrix, plot_comparison, plot_heads,  # noqa: E402
                           save_figure)

SAIDA = RAIZ / "Experiments" / "results" / "sprint3"
FIGURAS = SAIDA / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)

SEED = 123
with open(RAIZ / "the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def rotulo(token_id):
    """Texto legível de um token para os eixos dos gráficos."""
    txt = tokenizer.decode([token_id]).strip()
    return txt if txt else "␣"


def preparar_entrada(context_length, embedding_dim, amostra=0):
    """Pipeline da Sprint 2: texto -> token IDs -> token emb + positional emb.
    Retorna (x: (1, T, embedding_dim), tokens: lista de T strings)."""
    loader = create_dataloader(raw_text, batch_size=1, max_length=context_length,
                               stride=context_length, shuffle=False)
    for i, (inputs, _) in enumerate(loader):
        if i == amostra:
            break
    torch.manual_seed(SEED)
    tok_emb = nn.Embedding(VOCAB_SIZE, embedding_dim)
    pos_emb = nn.Embedding(context_length, embedding_dim)
    x = tok_emb(inputs) + pos_emb(torch.arange(context_length))
    return x.detach(), [rotulo(t) for t in inputs[0].tolist()]


def metricas_atencao(w, causal):
    """w: (..., T, T). Retorna (entropia_norm média, peso_max médio)."""
    T = w.shape[-1]
    w = w.detach().reshape(-1, T, T)
    visiveis = torch.arange(1, T + 1) if causal else torch.full((T,), T)
    ent = -(w * torch.log(w.clamp_min(1e-12))).sum(-1)              # (N, T)
    ent_norm = ent / torch.log(visiveis.float()).clamp_min(1e-12)   # linha 0 causal vira 0
    linhas = slice(1, None) if causal else slice(None)
    return (round(float(ent_norm[:, linhas].mean()), 4),
            round(float(w.max(-1).values[:, linhas].mean()), 4))


def n_params(modulo):
    return sum(p.numel() for p in modulo.parameters())


def diversidade_heads(w):
    """w: (heads, T, T). Diferença absoluta média entre pares de heads (0 = heads idênticas)."""
    w = w.detach()
    h = w.shape[0]
    if h < 2:
        return 0.0
    difs = [(w[i] - w[j]).abs().mean() for i in range(h) for j in range(i + 1, h)]
    return round(float(torch.stack(difs).mean()), 4)


resultados = []  # lista de dicts -> metricas.csv


def registrar(experimento, **kw):
    resultados.append({"experimento": experimento, **kw})


# ----------------------------------------------------------------------------
# Exp 1 — Attention com vs. sem escala (1/sqrt(d_k))
# ----------------------------------------------------------------------------
print("Exp 1: com vs. sem escala")
for d in (8, 64, 256, 768):
    x, tokens = preparar_entrada(context_length=8, embedding_dim=d)
    torch.manual_seed(SEED)
    sa = SelfAttention(d, d)                      # mesmos pesos nas duas execuções
    sa.scale = False
    _, w_sem = sa(x, return_weights=True)
    sa.scale = True
    _, w_com = sa(x, return_weights=True)
    for nome, w in (("sem escala", w_sem), ("com escala", w_com)):
        ent, pmax = metricas_atencao(w, causal=False)
        registrar("1_escala", d_k=d, escala=nome, entropia_norm=ent, peso_max=pmax)
    fig = plot_comparison([(f"Sem escala (d_k={d})", w_sem[0]),
                           (f"Com escala (d_k={d})", w_com[0])],
                          tokens, suptitle="Self-Attention: efeito da escala 1/√d_k",
                          annotate=(d <= 64))
    save_figure(fig, FIGURAS / f"exp1_escala_dk{d}.png")

# gráfico-resumo do Exp 1
fig, ax = plt.subplots(figsize=(6, 4))
for nome, marcador in (("sem escala", "o"), ("com escala", "s")):
    linhas = [r for r in resultados if r["experimento"] == "1_escala" and r["escala"] == nome]
    ax.plot([r["d_k"] for r in linhas], [r["peso_max"] for r in linhas], marker=marcador, label=nome)
ax.set_xscale("log", base=2)
ax.set_xlabel("d_k (dimensão de key/query)")
ax.set_ylabel("peso máximo médio por linha")
ax.set_title("Exp 1 — saturação do softmax")
ax.legend()
ax.grid(alpha=0.3)
save_figure(fig, FIGURAS / "exp1_resumo_saturacao.png")

# ----------------------------------------------------------------------------
# Exp 2 — Comportamento da máscara causal (Self vs. Causal) + teste de vazamento
# ----------------------------------------------------------------------------
print("Exp 2: máscara causal")
D, T = 64, 8
x, tokens = preparar_entrada(T, D)
torch.manual_seed(SEED)
sa = SelfAttention(D, D)
torch.manual_seed(SEED)
ca = CausalAttention(D, D, context_length=T)   # mesma seed -> mesmos Wq/Wk/Wv
out_sa, w_sa = sa(x, return_weights=True)
out_ca, w_ca = ca(x, return_weights=True)
fig = plot_comparison([("Self-Attention (sem máscara)", w_sa[0]),
                       ("Causal Attention (com máscara)", w_ca[0])],
                      tokens, suptitle="Efeito da máscara causal")
save_figure(fig, FIGURAS / "exp2_self_vs_causal.png")

# Vazamento: altera só os tokens a partir da posição 5 e compara as saídas das posições 0..4
x_mod = x.clone()
x_mod[:, 5:, :] += 5.0
with torch.no_grad():
    dif_self = float((sa(x)[:, :5] - sa(x_mod)[:, :5]).abs().max())
    dif_causal = float((ca(x)[:, :5] - ca(x_mod)[:, :5]).abs().max())
registrar("2_mascara", modelo="Self-Attention", soma_pesos_futuros=round(float(torch.triu(w_sa[0].detach(), 1).sum()), 4),
          vazamento_max_dif=round(dif_self, 6))
registrar("2_mascara", modelo="Causal Attention", soma_pesos_futuros=round(float(torch.triu(w_ca[0].detach(), 1).sum()), 4),
          vazamento_max_dif=round(dif_causal, 6))

# ----------------------------------------------------------------------------
# Exp 3 — Número de heads (d_out fixo = 256)
# ----------------------------------------------------------------------------
print("Exp 3: número de heads")
D, T = 256, 8
x, tokens = preparar_entrada(T, D)
for h in (1, 2, 4, 8):
    torch.manual_seed(SEED)
    mha = MultiHeadAttention(D, D, T, 0.0, h)
    _, w = mha(x, return_weights=True)
    ent, pmax = metricas_atencao(w, causal=True)
    registrar("3_num_heads", heads=h, head_dim=mha.head_dim, parametros=n_params(mha),
              entropia_norm=ent, peso_max=pmax, diversidade_heads=diversidade_heads(w[0]))
    fig = plot_heads(w[0], tokens, suptitle=f"Multi-Head Attention — {h} head(s), head_dim={mha.head_dim}")
    save_figure(fig, FIGURAS / f"exp3_heads_{h}.png")

# ----------------------------------------------------------------------------
# Exp 4 — Dimensão de cada head (num_heads = 4 fixo; varia d_out)
# ----------------------------------------------------------------------------
print("Exp 4: dimensão da head")
D, T, H = 256, 8, 4
x, tokens = preparar_entrada(T, D)
for d_out in (32, 64, 128, 256):
    torch.manual_seed(SEED)
    mha = MultiHeadAttention(D, d_out, T, 0.0, H)
    _, w = mha(x, return_weights=True)
    ent, pmax = metricas_atencao(w, causal=True)
    registrar("4_head_dim", d_out=d_out, head_dim=mha.head_dim, parametros=n_params(mha),
              entropia_norm=ent, peso_max=pmax)

# ----------------------------------------------------------------------------
# Exp 5 — Dimensão do embedding (num_heads = 4; d_out = d_emb)
# ----------------------------------------------------------------------------
print("Exp 5: dimensão do embedding")
T, H = 8, 4
for d_emb in (16, 64, 256, 768):
    x, tokens = preparar_entrada(T, d_emb)
    torch.manual_seed(SEED)
    mha = MultiHeadAttention(d_emb, d_emb, T, 0.0, H)
    _, w = mha(x, return_weights=True)
    ent, pmax = metricas_atencao(w, causal=True)
    registrar("5_dim_embedding", d_emb=d_emb, head_dim=mha.head_dim, parametros=n_params(mha),
              entropia_norm=ent, peso_max=pmax)

# ----------------------------------------------------------------------------
# Exp 6 — Self-Attention vs. Causal vs. Multi-Head (D=64, 4 heads)
# ----------------------------------------------------------------------------
print("Exp 6: Self vs. Causal vs. Multi-Head")
D, T, H = 64, 8, 4
x, tokens = preparar_entrada(T, D)
torch.manual_seed(SEED)
sa = SelfAttention(D, D)
torch.manual_seed(SEED)
ca = CausalAttention(D, D, T)
torch.manual_seed(SEED)
mha = MultiHeadAttention(D, D, T, 0.0, H)
_, w_sa = sa(x, return_weights=True)
_, w_ca = ca(x, return_weights=True)
_, w_mha = mha(x, return_weights=True)
for nome, modulo, w, causal in (("Self-Attention", sa, w_sa, False), ("Causal Attention", ca, w_ca, True),
                                (f"Multi-Head ({H} heads)", mha, w_mha, True)):
    ent, pmax = metricas_atencao(w, causal)
    registrar("6_comparacao", modelo=nome, parametros=n_params(modulo), entropia_norm=ent, peso_max=pmax)
fig = plot_comparison([("Self-Attention", w_sa[0]), ("Causal Attention", w_ca[0]),
                       ("Multi-Head: head 0", w_mha[0, 0]), ("Multi-Head: média das heads", w_mha[0].mean(0))],
                      tokens, suptitle="Self vs. Causal vs. Multi-Head", annotate=False)
save_figure(fig, FIGURAS / "exp6_self_causal_multihead.png")

# ----------------------------------------------------------------------------
# Exp 7 — Diferentes sequências de entrada (trechos e tamanhos de contexto)
# ----------------------------------------------------------------------------
print("Exp 7: sequências diferentes")
D, H = 64, 4
T = 8
for amostra in (0, 10, 50):                     # trechos diferentes do texto, T = 8
    x, tokens = preparar_entrada(T, D, amostra=amostra)
    torch.manual_seed(SEED)
    mha = MultiHeadAttention(D, D, T, 0.0, H)
    _, w = mha(x, return_weights=True)
    ent, pmax = metricas_atencao(w, causal=True)
    registrar("7_sequencias", tipo="trecho", amostra=amostra, T=T, entropia_norm=ent, peso_max=pmax)
    fig = plot_attention_matrix(w[0].mean(0), tokens, f"Média das heads — trecho {amostra}: {' '.join(tokens)}")
    save_figure(fig, FIGURAS / f"exp7_trecho{amostra}.png")

for T in (4, 8, 16, 32):                        # tamanhos de contexto diferentes, mesmo trecho inicial
    x, tokens = preparar_entrada(T, D, amostra=0)
    torch.manual_seed(SEED)
    mha = MultiHeadAttention(D, D, T, 0.0, H)
    _, w = mha(x, return_weights=True)
    ent, pmax = metricas_atencao(w, causal=True)
    registrar("7_sequencias", tipo="tamanho", amostra=0, T=T, entropia_norm=ent, peso_max=pmax)
    if T in (16, 32):
        fig = plot_attention_matrix(w[0].mean(0), tokens, f"Média das heads — contexto T={T}", annotate=False)
        save_figure(fig, FIGURAS / f"exp7_contexto{T}.png")

# ----------------------------------------------------------------------------
# Saída: metricas.csv + resumo_metricas.txt
# ----------------------------------------------------------------------------
colunas = ["experimento"] + sorted({k for r in resultados for k in r if k != "experimento"})
with open(SAIDA / "metricas.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=colunas)
    escritor.writeheader()
    escritor.writerows(resultados)

with open(SAIDA / "resumo_metricas.txt", "w", encoding="utf-8") as f:
    for exp in sorted({r["experimento"] for r in resultados}):
        linhas = [r for r in resultados if r["experimento"] == exp]
        cols = [c for c in colunas if c != "experimento" and any(c in r for r in linhas)]
        larg = [max(len(c), *(len(str(r.get(c, ""))) for r in linhas)) for c in cols]
        f.write(f"{exp}\n" + "-" * len(exp) + "\n")
        f.write("  ".join(c.ljust(w) for c, w in zip(cols, larg)) + "\n")
        f.write("  ".join("-" * w for w in larg) + "\n")
        for r in linhas:
            f.write("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, larg)) + "\n")
        f.write("\n")

print(f"\nConcluído: {len(resultados)} linhas de métricas.")
print(f"Figuras em: {FIGURAS}")
print(f"Métricas em: {SAIDA / 'metricas.csv'}")
