"""
Experimentos da Sprint 4 — arquitetura GPT e geração de texto.

Como rodar (a partir da RAIZ do repositório):
    python Experiments/experiments_sprint4.py

Saídas (em Experiments/results/sprint4/):
    figuras/*.png         um gráfico por experimento
    metricas.csv          todas as métricas (formato longo)
    resumo_metricas.txt   tabelas em texto simples, para consultar na análise

IMPORTANTE (mesma ressalva da Sprint 3): os modelos NÃO são treinados (pesos
aleatórios, seed fixa). Por isso:
  - Experimentos 1 a 5 (parâmetros, tempo, memória) NÃO dependem do treino:
    o custo computacional é o mesmo com ou sem treino.
  - Experimento 6 (dropout) mostra o mecanismo (aleatoriedade no treino).
  - Experimento 7 (temperatura / top-k) mostra o efeito sobre a distribuição de
    probabilidades, mas NÃO a qualidade do texto (um modelo sem treino gera
    texto sem sentido de qualquer forma).
Os Token IDs de entrada são aleatórios (seed fixa): para um modelo sem treino
o conteúdo do texto não altera custo nem estrutura.

Métricas de custo computacional:
    parametros / MB     determinísticos (MB = float32, 4 bytes por parâmetro)
    gflops              nº estimado de operações (GFLOPs) de UM forward, contando só as
                        multiplicações de matrizes (determinístico, ver flops_forward)
    forward_ms          tempo do modelo completo (inclui a projeção final para 50257 tokens)
    blocos_ms           tempo SÓ dos Transformer Blocks (isola o efeito da arquitetura)
Os TEMPOS dependem da máquina (CPU/GPU, nº de threads): compare tendências entre
configurações, não valores absolutos. Diferenças de ~10% entre tempos são ruído.
Compare tempos DENTRO de um mesmo experimento (a mesma configuração-base pode medir
valores diferentes em experimentos diferentes por oscilação da máquina).
"""

import csv
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # salva em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "Codes" / "SPRINT_3"))
sys.path.insert(0, str(RAIZ / "Codes" / "SPRINT_4"))

from gpt_model import GPTModel, contar_parametros  # noqa: E402
from generate import generate  # noqa: E402

SAIDA = RAIZ / "Experiments" / "results" / "sprint4"
FIGURAS = SAIDA / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)

SEED = 123
VOCAB = 50257
# Configuração-base (menor que o GPT-2 small para os experimentos rodarem rápido).
BASE = {"vocab_size": VOCAB, "context_length": 256, "emb_dim": 256, "n_heads": 8,
        "n_layers": 4, "drop_rate": 0.0, "qkv_bias": False}
BATCH, T_PADRAO = 4, 128

linhas_csv = []   # (experimento, configuracao, metrica, valor)
resumo = []       # blocos de texto para resumo_metricas.txt


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def registrar(exp, config, **metricas):
    for nome, valor in metricas.items():
        linhas_csv.append((exp, config, nome, valor))


def tabela(titulo, cabecalho, linhas):
    larg = [max(len(str(x)) for x in col) for col in zip(cabecalho, *linhas)]
    fmt = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, larg))
    bloco = [titulo, "-" * len(titulo), fmt(cabecalho), fmt(["-" * w for w in larg])]
    bloco += [fmt(r) for r in linhas]
    resumo.append("\n".join(bloco))
    print(resumo[-1] + "\n")


def novo_modelo(**mudancas):
    cfg = {**BASE, **mudancas}
    torch.manual_seed(SEED)
    return GPTModel(cfg), cfg


def ids_aleatorios(b, t):
    g = torch.Generator().manual_seed(SEED)
    return torch.randint(0, VOCAB, (b, t), generator=g)


def tempo_forward_ms(model, ids, repeticoes=9, aquecimento=2):
    """Menor tempo de um forward em modo de inferência (ms). O mínimo é mais robusto
    que a média a interferências de outros processos na máquina."""
    model.eval()
    with torch.no_grad():
        for _ in range(aquecimento):
            model(ids)
        ts = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            model(ids)
            ts.append((time.perf_counter() - t0) * 1000)
    return min(ts)


def tempo_blocos_ms(model, b, t, repeticoes=15, aquecimento=3):
    """Menor tempo (ms) dos Transformer Blocks sozinhos (sem embeddings nem head)."""
    model.eval()
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(b, t, model.tok_emb.embedding_dim, generator=g)
    with torch.no_grad():
        for _ in range(aquecimento):
            model.trf_blocks(x)
        ts = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            model.trf_blocks(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return min(ts)


def flops_forward(cfg, b, t):
    """GFLOPs de um forward (só multiplicações de matrizes; 1 multiplicação+soma = 2 FLOPs).
    Por token e por bloco: projeções Q,K,V,out = 8*d^2 ; scores e soma ponderada = 4*T*d ;
    FeedForward = 4*d*ff. Mais a projeção final (head) = 2*d*V por token."""
    d, ff = cfg["emb_dim"], cfg.get("ff_dim", 4 * cfg["emb_dim"])
    por_bloco = 8 * d * d + 4 * t * d + 4 * d * ff
    por_token = cfg["n_layers"] * por_bloco + 2 * d * cfg["vocab_size"]
    return b * t * por_token / 1e9


def mb(n_params):
    return n_params * 4 / 1024 ** 2  # float32


def grafico_linha(nome, x, ys, xlabel, titulo, ylabels, logx=False, logy=False):
    fig, axs = plt.subplots(1, len(ys), figsize=(5 * len(ys), 3.8))
    axs = [axs] if len(ys) == 1 else axs
    for ax, y, yl in zip(axs, ys, ylabels):
        if isinstance(y, dict):
            for rot, serie in y.items():
                ax.plot(x, serie, marker="o", label=rot)
            ax.legend()
        else:
            ax.plot(x, y, marker="o")
        ax.set_xlabel(xlabel); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        if logx: ax.set_xscale("log", base=2)
        if logy: ax.set_yscale("log", base=2)
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(FIGURAS / nome, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Função comum dos experimentos 1 a 4 (arquitetura): mede custo de uma configuração
# ----------------------------------------------------------------------------
def medir(exp, rotulo, t=T_PADRAO, **mudancas):
    m, cfg = novo_modelo(**mudancas)
    ids = ids_aleatorios(BATCH, t)
    p = contar_parametros(m)
    fwd, blk = tempo_forward_ms(m, ids), tempo_blocos_ms(m, BATCH, t)
    gf = flops_forward(cfg, BATCH, t)
    registrar(exp, rotulo, parametros=p["total"], parametros_blocos=p["blocos"],
              parametros_atencao=p["atencao"], parametros_feedforward=p["feedforward"],
              mb_fp32=round(mb(p["total"]), 1), gflops=round(gf, 2),
              forward_ms=round(fwd, 1), blocos_ms=round(blk, 1))
    return p, cfg, fwd, blk, gf


# ----------------------------------------------------------------------------
# Experimento 1 — número de Transformer Blocks (n_layers)
# ----------------------------------------------------------------------------
def exp1_n_layers():
    valores, linhas, params, fwds, blks = [1, 2, 4, 8, 12], [], [], [], []
    for n in valores:
        p, _, fwd, blk, gf = medir("exp1_n_layers", f"n_layers={n}", n_layers=n)
        linhas.append([n, f"{p['total']:,}", f"{p['blocos']:,}", f"{mb(p['total']):.1f}",
                       f"{gf:.1f}", f"{fwd:.0f}", f"{blk:.0f}"])
        params.append(p["total"] / 1e6); fwds.append(fwd); blks.append(blk)
    tabela("1_n_layers (emb=256, heads=8, ff=4x, T=128, batch=4)",
           ["n_layers", "parametros", "param_blocos", "MB(fp32)", "GFLOPs", "forward_ms", "blocos_ms"], linhas)
    grafico_linha("exp1_n_layers.png", valores, [params, {"modelo completo": fwds, "só os blocos": blks}],
                  "nº de Transformer Blocks", "Exp 1 — número de blocos",
                  ["parâmetros (milhões)", "tempo do forward (ms)"])


# ----------------------------------------------------------------------------
# Experimento 2 — número de heads (n_heads)
# ----------------------------------------------------------------------------
def exp2_n_heads():
    valores, linhas, fwds, blks = [1, 2, 4, 8, 16], [], [], []
    for h in valores:
        p, cfg, fwd, blk, gf = medir("exp2_n_heads", f"n_heads={h}", n_heads=h)
        linhas.append([h, cfg["emb_dim"] // h, f"{p['total']:,}", f"{gf:.1f}", f"{fwd:.0f}", f"{blk:.0f}"])
        fwds.append(fwd); blks.append(blk)
    tabela("2_n_heads (emb=256, 4 blocos, T=128, batch=4)",
           ["heads", "head_dim", "parametros", "GFLOPs", "forward_ms", "blocos_ms"], linhas)
    grafico_linha("exp2_n_heads.png", valores,
                  [[256 // h for h in valores], {"modelo completo": fwds, "só os blocos": blks}],
                  "nº de heads", "Exp 2 — número de heads (parâmetros e GFLOPs constantes)",
                  ["head_dim = emb_dim / heads", "tempo do forward (ms)"])


# ----------------------------------------------------------------------------
# Experimento 3 — dimensão dos embeddings (emb_dim)
# ----------------------------------------------------------------------------
def exp3_emb_dim():
    valores, linhas, params, fwds, blks = [64, 128, 256, 512], [], [], [], []
    for d in valores:
        p, _, fwd, blk, gf = medir("exp3_emb_dim", f"emb_dim={d}", emb_dim=d)
        emb_total = p["tok_emb"] + p["pos_emb"] + p["out_head"]
        linhas.append([d, f"{p['total']:,}", f"{p['blocos']:,}", f"{100 * emb_total / p['total']:.1f}%",
                       f"{gf:.1f}", f"{fwd:.0f}", f"{blk:.0f}"])
        params.append(p["total"] / 1e6); fwds.append(fwd); blks.append(blk)
    tabela("3_emb_dim (heads=8, 4 blocos, ff=4x, T=128, batch=4)",
           ["emb_dim", "parametros", "param_blocos", "%tok_emb+pos_emb+head", "GFLOPs", "forward_ms", "blocos_ms"],
           linhas)
    grafico_linha("exp3_emb_dim.png", valores, [params, {"modelo completo": fwds, "só os blocos": blks}],
                  "emb_dim", "Exp 3 — dimensão dos embeddings",
                  ["parâmetros (milhões)", "tempo do forward (ms)"], logx=True)


# ----------------------------------------------------------------------------
# Experimento 4 — dimensão da FeedForward (ff_dim = mult * emb_dim)
# ----------------------------------------------------------------------------
def exp4_ff_dim():
    valores, linhas, params, fwds, blks = [1, 2, 4, 8], [], [], [], []
    for mult in valores:
        p, cfg, fwd, blk, gf = medir("exp4_ff_dim", f"ff={mult}x", ff_dim=mult * BASE["emb_dim"])
        pct_ff = 100 * p["feedforward"] / p["blocos"]
        linhas.append([f"{mult}x", cfg["ff_dim"], f"{p['total']:,}", f"{pct_ff:.1f}%",
                       f"{gf:.1f}", f"{fwd:.0f}", f"{blk:.0f}"])
        params.append(p["total"] / 1e6); fwds.append(fwd); blks.append(blk)
    tabela("4_ff_dim (emb=256, heads=8, 4 blocos, T=128, batch=4)",
           ["ff_mult", "ff_dim", "parametros", "%FF_nos_blocos", "GFLOPs", "forward_ms", "blocos_ms"], linhas)
    grafico_linha("exp4_ff_dim.png", valores, [params, {"modelo completo": fwds, "só os blocos": blks}],
                  "ff_dim / emb_dim", "Exp 4 — dimensão da FeedForward",
                  ["parâmetros (milhões)", "tempo do forward (ms)"])


# ----------------------------------------------------------------------------
# Experimento 5 — tamanho da sequência (T): custo quadrático da atenção
# ----------------------------------------------------------------------------
def exp5_contexto():
    valores, linhas, blks, att_mbs = [16, 32, 64, 128, 256, 512], [], [], []
    for t in valores:
        p, cfg, fwd, blk, gf = medir("exp5_contexto", f"T={t}", t=t, context_length=512)
        # matrizes de atenção de UM bloco: (batch, heads, T, T) em float32
        att_mb = BATCH * cfg["n_heads"] * t * t * 4 / 1024 ** 2
        # fatia do custo dos blocos que é "scores + soma ponderada" (a parte quadrática em T)
        d = cfg["emb_dim"]
        pct_quad = 100 * (4 * t * d) / (8 * d * d + 4 * t * d + 4 * d * 4 * d)
        registrar("exp5_contexto", f"T={t}", matriz_atencao_mb=round(att_mb, 2),
                  pct_flops_blocos_em_scores=round(pct_quad, 1),
                  blocos_ms_por_token=round(blk / (BATCH * t), 4))
        linhas.append([t, f"{gf:.1f}", f"{fwd:.0f}", f"{blk:.0f}", f"{blk / (BATCH * t):.3f}",
                       f"{att_mb:.2f}", f"{pct_quad:.1f}%"])
        blks.append(blk); att_mbs.append(att_mb)
    tabela("5_contexto (emb=256, heads=8, 4 blocos, batch=4, context_length=512)",
           ["T", "GFLOPs", "forward_ms", "blocos_ms", "blocos_ms/token", "matriz_atencao_MB(1 bloco)",
            "%FLOPs_blocos_em_scores"], linhas)
    grafico_linha("exp5_contexto.png", valores, [blks, att_mbs], "T (nº de tokens)",
                  "Exp 5 — tamanho da sequência",
                  ["tempo dos blocos (ms)", "matrizes de atenção, 1 bloco (MB)"], logx=True, logy=True)


# ----------------------------------------------------------------------------
# Experimento 6 — dropout (modo treino vs. inferência)
# ----------------------------------------------------------------------------
def exp6_dropout():
    ids = ids_aleatorios(BATCH, 32)
    valores, linhas, difs = [0.0, 0.1, 0.3, 0.5], [], []
    for p_drop in valores:
        m, _ = novo_modelo(drop_rate=p_drop, n_layers=2)
        with torch.no_grad():
            m.eval(); a_eval, b_eval = m(ids), m(ids)
            m.train(); a_tr, b_tr = m(ids), m(ids)
        dif_eval = (a_eval - b_eval).abs().mean().item()
        dif_tr = (a_tr - b_tr).abs().mean().item()
        registrar("exp6_dropout", f"drop={p_drop}", dif_media_eval=round(dif_eval, 6),
                  dif_media_treino=round(dif_tr, 6))
        linhas.append([p_drop, f"{dif_eval:.6f}", f"{dif_tr:.4f}"])
        difs.append(dif_tr)
    tabela("6_dropout (2 blocos; |logits_a - logits_b| médio, mesma entrada, 2 passadas)",
           ["drop_rate", "dif_modo_eval", "dif_modo_treino"], linhas)
    grafico_linha("exp6_dropout.png", valores, [difs], "drop_rate",
                  "Exp 6 — variação entre duas passadas (modo treino)", ["|Δ logits| médio"])


# ----------------------------------------------------------------------------
# Experimento 7 — parâmetros de geração: temperatura e top-k
# ----------------------------------------------------------------------------
def entropia(probs):
    p = probs[probs > 0]
    return -(p * p.log()).sum().item()


def exp7_geracao():
    m, cfg = novo_modelo(n_layers=4)
    m.eval()
    prompt = ids_aleatorios(1, 8)
    with torch.no_grad():
        logits = m(prompt)[0, -1]
    print(f"  (logits do modelo sem treino: desvio-padrão = {logits.std():.3f}, "
          f"max-min = {(logits.max() - logits.min()):.3f})\n")
    registrar("exp7_geracao", "logits_sem_treino", desvio_padrao=round(logits.std().item(), 4))
    base = torch.softmax(logits, dim=-1)

    # (a) efeito da temperatura e do top-k sobre a distribuição do próximo token
    #   entropia_norm : entropia / log(V)        (1 = uniforme sobre TODO o vocabulário)
    #   entropia_suporte: entropia / log(k)      (1 = uniforme entre os candidatos que sobraram)
    #   massa_original: soma das probabilidades ORIGINAIS (T=1) dos k tokens mantidos
    linhas, temps, curvas = [], [0.1, 0.5, 1.0, 2.0, 5.0], {}
    for k in [None, 50, 10, 1]:
        curvas[k] = []
        l = logits.clone()
        if k is not None:
            l[l < torch.topk(l, k).values[-1]] = float("-inf")
        massa = base[l > float("-inf")].sum().item()
        suporte = int((l > float("-inf")).sum())
        for T in temps:
            probs = torch.softmax(l / T, dim=-1)
            e = entropia(probs)
            e_norm = abs(e) / math.log(probs.numel())
            e_sup = abs(e) / math.log(suporte) if suporte > 1 else 0.0
            pmax = probs.max().item()
            registrar("exp7_geracao", f"top_k={k}|T={T}", entropia_norm=round(e_norm, 4),
                      entropia_suporte=round(e_sup, 4), prob_max=round(pmax, 5),
                      tokens_candidatos=suporte, massa_original_mantida=round(massa, 5))
            linhas.append([k if k else "-", T, f"{e_norm:.4f}", f"{e_sup:.4f}", f"{pmax:.5f}",
                           suporte, f"{100 * massa:.3f}%"])
            curvas[k].append(e_norm)
    tabela("7a_distribuicao_proximo_token (modelo sem treino; V=50257)",
           ["top_k", "T", "entropia/log(V)", "entropia/log(k)", "prob_max", "candidatos",
            "massa_original_mantida"], linhas)

    # (b) diversidade do 1º token: 100 amostras com seeds diferentes
    linhas2 = []
    for nome, kw in [("greedy", dict(temperature=0.0)),
                     ("T=1.0", dict(temperature=1.0)),
                     ("T=1.0, top_k=50", dict(temperature=1.0, top_k=50)),
                     ("T=1.0, top_k=5", dict(temperature=1.0, top_k=5)),
                     ("T=0.1", dict(temperature=0.1)),
                     ("T=5.0", dict(temperature=5.0))]:
        primeiros = []
        for s in range(100):
            torch.manual_seed(1000 + s)
            primeiros.append(generate(m, prompt, 1, cfg["context_length"], **kw)[0, -1].item())
        distintos = len(set(primeiros))
        registrar("exp7_geracao", f"1o_token|{nome}", tokens_distintos_em_100=distintos)
        linhas2.append([nome, f"{distintos}/100"])
    tabela("7b_diversidade (100 sorteios do 1º token novo, seeds diferentes, mesmo prompt)",
           ["estrategia", "tokens_distintos"], linhas2)

    # figura: entropia normalizada x temperatura, uma linha por top-k
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, ys in curvas.items():
        ax.plot(temps, ys, marker="o", label=f"top_k={k}" if k else "sem top-k")
    ax.set_xscale("log"); ax.set_xlabel("temperatura"); ax.set_ylabel("entropia / log(V)")
    ax.set_title("Exp 7 — temperatura e top-k (modelo sem treino)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURAS / "exp7_temperatura_topk.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------
def aquecer(segundos=8):
    """Roda o modelo-base por alguns segundos antes de medir (a CPU/alocador de memória
    costumam estar mais lentos nas primeiras chamadas, o que distorceria os 1os experimentos)."""
    m, _ = novo_modelo()
    ids = ids_aleatorios(BATCH, T_PADRAO)
    fim = time.perf_counter() + segundos
    while time.perf_counter() < fim:
        tempo_forward_ms(m, ids, repeticoes=1, aquecimento=0)
        tempo_blocos_ms(m, BATCH, T_PADRAO, repeticoes=1, aquecimento=0)


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} | threads: {torch.get_num_threads()} | seed {SEED}\n")
    aquecer()
    for fn in (exp1_n_layers, exp2_n_heads, exp3_emb_dim, exp4_ff_dim,
               exp5_contexto, exp6_dropout, exp7_geracao):
        print(f">>> {fn.__name__}")
        fn()

    with open(SAIDA / "metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experimento", "configuracao", "metrica", "valor"])
        w.writerows(linhas_csv)
    cab = (f"PyTorch {torch.__version__} | threads: {torch.get_num_threads()} | seed {SEED}\n"
           "Tempos dependem da máquina; compare tendências, não valores absolutos.\n")
    (SAIDA / "resumo_metricas.txt").write_text(cab + "\n" + "\n\n".join(resumo) + "\n", encoding="utf-8")
    print(f"Resultados salvos em {SAIDA}")
