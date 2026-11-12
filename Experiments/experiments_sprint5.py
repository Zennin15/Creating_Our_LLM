"""
Experimentos da Sprint 5 — treinamento, avaliação e geração de texto.

Como rodar (a partir da RAIZ do repositório):
    python Experiments/experiments_sprint5.py                 # completo
    python Experiments/experiments_sprint5.py --preset rapido # metade do custo, para testar
    python Experiments/experiments_sprint5.py --preset smoke  # segundos; só confere o pipeline

Saídas (em Experiments/results/sprint5/):
    figuras/*.png          um gráfico por experimento
    metricas.csv           todas as métricas (formato longo)
    resumo_metricas.txt    tabelas em texto simples, para consultar na análise
    amostras_geracao.txt   TODOS os textos gerados no experimento 6 (leia-os!)
    curvas/*.csv           loss de treino/validação por época de cada treino
    modelo_treinado.pt     pesos do modelo-base (10 épocas), para a demonstração de load
    (o preset "smoke" usa um tokenizer de bytes e um modelo minúsculo: serve só para
     conferir que tudo roda; NÃO use seus resultados no relatório.)

Dados: the-verdict.txt (os mesmos dados do livro), divididos 90% treino / 10% validação
NO TEXTO, em ordem (validação = final do conto, nunca visto no treino).

Experimentos (muda-se UMA coisa por vez em relação à base):
    0  estado inicial do modelo (loss ~ log V, perplexity ~ V)
    1  número de épocas   (UM treino longo, com avaliação a cada época)
    2  taxa de aprendizado
    3  tamanho do lote
    4  dropout (regularização) -> efeito sobre o sobreajuste
    5  custo computacional (tempo, tokens/s) — coletado em todos os treinos
    6  decodificação: greedy x amostragem x temperatura x top-k, no mesmo prompt, em
       três modelos (sem treino, treinado, sobreajustado)

Cuidados com a leitura:
  - Os dados são MUITO pequenos (~5 mil tokens): o sobreajuste aparece rápido. Isso é o
    comportamento esperado, não um defeito do código.
  - Seed fixa (123) e UMA execução por configuração: diferenças pequenas (~0.1 de loss)
    podem ser ruído de inicialização/ordem dos lotes.
  - Tempos dependem da máquina; compare tendências DENTRO deste experimento.
  - A loss é avaliada com TODOS os lotes (eval_iter=None) no fim de cada época.
"""

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RAIZ = Path(__file__).resolve().parents[1]
for sprint in ("SPRINT_2", "SPRINT_3", "SPRINT_4", "SPRINT_5"):
    sys.path.insert(0, str(RAIZ / "Codes" / sprint))

from gpt_model import GPTModel, contar_parametros  # noqa: E402
from loss import avaliar_modelo, perplexity, loss_inicial_esperada  # noqa: E402
from train import criar_loaders, treinar_modelo, novo_otimizador, escolher_dispositivo  # noqa: E402
from decoding_strategies import comparar_estrategias  # noqa: E402
from checkpoint import salvar_checkpoint  # noqa: E402

SEED = 123
PROMPT = "Every effort moves you"

PRESETS = {
    # ep_base: épocas do modelo-base; ep_total: épocas do treino longo (exp. 1 e 6)
    "completo": dict(ep_base=10, ep_total=30, ep_exp=10, ep_drop=20,
                     lrs=[1e-5, 1e-4, 5e-4, 2e-3], lotes=[1, 2, 4, 8], drops=[0.0, 0.1, 0.3],
                     modelo=dict(context_length=256, emb_dim=256, n_heads=8, n_layers=4),
                     max_new_tokens=50, seeds=(1, 2, 3, 4, 5)),
    "rapido": dict(ep_base=5, ep_total=15, ep_exp=5, ep_drop=10,
                   lrs=[1e-4, 5e-4, 2e-3], lotes=[2, 4], drops=[0.0, 0.3],
                   modelo=dict(context_length=128, emb_dim=128, n_heads=4, n_layers=2),
                   max_new_tokens=40, seeds=(1, 2, 3)),
    "smoke": dict(ep_base=2, ep_total=4, ep_exp=2, ep_drop=3,
                  lrs=[5e-4, 2e-3], lotes=[2, 4], drops=[0.0, 0.3],
                  modelo=dict(context_length=32, emb_dim=32, n_heads=4, n_layers=2),
                  max_new_tokens=15, seeds=(1, 2)),
}
BASE_LR, BASE_LOTE, BASE_WD = 5e-4, 2, 0.1

linhas_csv = []   # (experimento, configuracao, metrica, valor)
resumo = []       # blocos de texto para resumo_metricas.txt


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
class ByteTokenizer:
    """Só para o preset smoke (não precisa baixar o vocabulário do GPT-2)."""
    n_vocab = 256

    def encode(self, texto, allowed_special=None):
        return list(texto.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


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


def f3(x):
    return f"{x:.3f}"


def novo_modelo(cfg_modelo, device):
    torch.manual_seed(SEED)                 # mesma inicialização em todos os treinos
    return GPTModel(cfg_modelo).to(device)


def treinar(nome, cfg_modelo, texto, tok, device, epocas, lr=BASE_LR, lote=BASE_LOTE,
            wd=BASE_WD, model=None, optimizer=None, historico=None, loaders=None):
    """
    Treina (ou continua treinando) e devolve um registro com curvas, perdas finais e custo.
    Avalia com TODOS os lotes ao fim de cada época (eval_freq = nº de lotes de treino).
    """
    train_loader, val_loader = loaders or criar_loaders(
        texto, tok, lote, cfg_modelo["context_length"], seed=SEED)
    if model is None:
        model = novo_modelo(cfg_modelo, device)
        optimizer = novo_otimizador(model, lr, wd)
    t0 = time.perf_counter()
    h = treinar_modelo(model, train_loader, val_loader, optimizer, device, epocas,
                       eval_freq=len(train_loader), eval_iter=None, tokenizer=None,
                       historico=historico, verbose=False)
    segundos = time.perf_counter() - t0
    n_passos = h["passos"][-1]
    return {"nome": nome, "model": model, "optimizer": optimizer, "h": h,
            "n_lotes": len(train_loader), "n_passos": n_passos, "segundos": h["segundos"],
            "wall": segundos, "lr": lr, "lote": lote, "wd": wd,
            "train": h["train_loss"][-1], "val": h["val_loss"][-1],
            "loaders": (train_loader, val_loader)}


def curva_por_epoca(rec):
    """(épocas, train_loss, val_loss) com o ponto 0 = estado inicial."""
    h = rec["h"]
    return list(range(len(h["passos"]))), h["train_loss"], h["val_loss"]


def salvar_curva(rec, pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    ep, tl, vl = curva_por_epoca(rec)
    with open(pasta / f"{rec['nome']}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoca", "passo", "train_loss", "val_loss", "train_ppl", "val_ppl"])
        for e, p, a, b in zip(ep, rec["h"]["passos"], tl, vl):
            w.writerow([e, p, f"{a:.5f}", f"{b:.5f}", f"{perplexity(a):.3f}", f"{perplexity(b):.3f}"])


def custo(rec):
    tokens = rec["h"]["tokens_vistos"][-1]
    return (rec["segundos"], 1000 * rec["segundos"] / max(rec["n_passos"], 1),
            tokens / max(rec["segundos"], 1e-9))


def plot_curvas(recs, titulo, caminho, rotulos):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for rec, rot in zip(recs, rotulos):
        ep, tl, vl = curva_por_epoca(rec)
        l, = axs[0].plot(ep, tl, label=rot)
        axs[1].plot(ep, vl, label=rot, color=l.get_color())
    axs[0].set_title("loss de treino")
    axs[1].set_title("loss de validação")
    for ax in axs:
        ax.set_xlabel("épocas")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("loss (cross-entropy)")
    axs[1].legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="completo")
    ap.add_argument("--dados", default=str(RAIZ / "the-verdict.txt"))
    args = ap.parse_args()
    P = PRESETS[args.preset]
    smoke = args.preset == "smoke"

    saida = RAIZ / "Experiments" / "results" / ("sprint5_smoke" if smoke else "sprint5")
    figuras, curvas = saida / "figuras", saida / "curvas"
    figuras.mkdir(parents=True, exist_ok=True)

    if smoke:
        tok, vocab = ByteTokenizer(), 256
    else:
        import tiktoken
        tok, vocab = tiktoken.get_encoding("gpt2"), 50257
    device = escolher_dispositivo()
    texto = Path(args.dados).read_text(encoding="utf-8")
    cfg = {"vocab_size": vocab, "drop_rate": 0.0, "qkv_bias": False, **P["modelo"]}
    n_tok = len(tok.encode(texto))
    print(f"Preset: {args.preset} | dispositivo: {device} | tokens no texto: {n_tok:,}")
    print(f"Modelo-base: {cfg}\n")

    # ---------------------------------------------------------------- Exp. 0
    m0 = novo_modelo(cfg, device)
    tl_loader, vl_loader = criar_loaders(texto, tok, BASE_LOTE, cfg["context_length"], seed=SEED)
    tl0, vl0 = avaliar_modelo(m0, tl_loader, vl_loader, device, None)
    lnv = loss_inicial_esperada(vocab)
    registrar("exp0", "estado_inicial", train_loss=tl0, val_loss=vl0, train_ppl=perplexity(tl0),
              val_ppl=perplexity(vl0), log_vocab=lnv, vocab=vocab, tokens_texto=n_tok,
              lotes_treino=len(tl_loader), lotes_val=len(vl_loader),
              parametros=contar_parametros(m0)["total"])
    tabela("EXPERIMENTO 0 - Estado inicial (pesos aleatórios, seed 123)",
           ["conjunto", "loss", "perplexity", "log(V)", "V"],
           [["treino", f3(tl0), f"{perplexity(tl0):.1f}", f3(lnv), vocab],
            ["validação", f3(vl0), f"{perplexity(vl0):.1f}", f3(lnv), vocab]])
    resumo.append(f"Parâmetros do modelo-base: {contar_parametros(m0)['total']:,} | "
                  f"lotes de treino: {len(tl_loader)} | lotes de validação: {len(vl_loader)}")
    del m0

    # ---------------------------------------------------------------- Exp. 1 (épocas)
    # Um treino longo em duas fases. A fase 1 (ep_base épocas) vira o MODELO-BASE; a fase 2
    # continua o MESMO treino (mesmo otimizador, histórico contínuo) até ep_total.
    print("Exp. 1: treino longo (épocas)...")
    rec = treinar("exp1_epocas", cfg, texto, tok, device, P["ep_base"])
    base = {k: v for k, v in rec.items() if k not in ("model", "optimizer", "loaders")}
    base["h"] = copy.deepcopy(rec["h"])
    modelo_base = copy.deepcopy(rec["model"])             # snapshot: modelo-base
    if not smoke:
        salvar_checkpoint(saida / "modelo_treinado.pt", modelo_base, None, cfg,
                          {"epochs": P["ep_base"], "batch_size": BASE_LOTE, "lr": BASE_LR,
                           "weight_decay": BASE_WD, "optimizer": "AdamW", "seed": SEED,
                           "dados": "the-verdict.txt", "train_ratio": 0.9},
                          history=base["h"], extra={"origem": "experiments_sprint5.py"})
    rec = treinar("exp1_epocas", cfg, texto, tok, device, P["ep_total"] - P["ep_base"],
                  model=rec["model"], optimizer=rec["optimizer"], historico=rec["h"],
                  loaders=rec["loaders"], lr=BASE_LR, lote=BASE_LOTE)
    modelo_longo = rec["model"]                           # modelo após ep_total épocas
    salvar_curva(rec, curvas)
    ep, tl, vl = curva_por_epoca(rec)
    melhor = min(range(len(vl)), key=lambda i: vl[i])
    marcos = sorted({0, 1, max(P["ep_base"] // 2, 1), P["ep_base"], (P["ep_base"] + P["ep_total"]) // 2,
                     P["ep_total"]})
    linhas = []
    for e in marcos:
        linhas.append([e, f3(tl[e]), f3(vl[e]), f3(vl[e] - tl[e]), f"{perplexity(tl[e]):.1f}",
                       f"{perplexity(vl[e]):.1f}"])
        registrar("exp1", f"epoca_{e}", train_loss=tl[e], val_loss=vl[e], gap=vl[e] - tl[e],
                  train_ppl=perplexity(tl[e]), val_ppl=perplexity(vl[e]))
    seg, ms_passo, tok_s = custo(rec)
    registrar("exp1", "total", melhor_epoca_val=melhor, menor_val_loss=vl[melhor],
              segundos=seg, ms_por_passo=ms_passo, tokens_por_s=tok_s)
    tabela(f"EXPERIMENTO 1 - Número de épocas (lr {BASE_LR}, lote {BASE_LOTE}, {P['ep_total']} épocas)",
           ["época", "loss treino", "loss val", "val - treino", "ppl treino", "ppl val"], linhas)
    resumo.append(f"Menor loss de validação: {vl[melhor]:.3f} na época {melhor} "
                  f"(de {P['ep_total']}). Depois dela a validação sobe se há sobreajuste.\n"
                  f"Custo: {seg:.0f} s no total | {ms_passo:.0f} ms/passo | {tok_s:,.0f} tokens/s")
    print(resumo[-1] + "\n")
    plot_curvas([rec], f"Exp. 1 - Épocas (lr {BASE_LR}, lote {BASE_LOTE})",
                figuras / "exp1_epocas.png", ["treino longo"])
    # ponto onde o modelo-base para: aviso no gráfico
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, tl, label="treino")
    ax.plot(ep, vl, "--", label="validação")
    ax.axvline(melhor, color="gray", linestyle=":", label=f"menor val (época {melhor})")
    ax.set_xlabel("épocas")
    ax.set_ylabel("loss")
    ax.set_title("Exp. 1 - sobreajuste: treino cai, validação sobe")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figuras / "exp1_sobreajuste.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- Exp. 2 (lr)
    print("Exp. 2: taxa de aprendizado...")
    recs, rot, linhas = [], [], []
    for lr in P["lrs"]:
        if lr == BASE_LR and P["ep_exp"] == P["ep_base"]:
            r = {**base, "h": base["h"]}                 # mesma config do modelo-base
        else:
            r = treinar(f"exp2_lr_{lr:g}", cfg, texto, tok, device, P["ep_exp"], lr=lr)
        r["nome"] = f"exp2_lr_{lr:g}"
        salvar_curva(r, curvas)
        recs.append(r)
        rot.append(f"lr={lr:g}")
        e = P["ep_exp"]
        tl_, vl_ = r["h"]["train_loss"][e], r["h"]["val_loss"][e]
        seg, msp, tks = custo(r)
        linhas.append([f"{lr:g}", f3(tl_), f3(vl_), f3(min(r['h']['val_loss'])), f"{seg:.0f}"])
        registrar("exp2", f"lr_{lr:g}", train_loss=tl_, val_loss=vl_,
                  menor_val_loss=min(r["h"]["val_loss"]), segundos=seg, ms_por_passo=msp)
    tabela(f"EXPERIMENTO 2 - Taxa de aprendizado ({P['ep_exp']} épocas, lote {BASE_LOTE})",
           ["lr", "loss treino (final)", "loss val (final)", "menor loss val", "segundos"], linhas)
    plot_curvas(recs, f"Exp. 2 - Taxa de aprendizado ({P['ep_exp']} épocas)",
                figuras / "exp2_taxa_aprendizado.png", rot)

    # ---------------------------------------------------------------- Exp. 3 (lote)
    print("Exp. 3: tamanho do lote...")
    recs, rot, linhas = [], [], []
    for lote in P["lotes"]:
        if lote == BASE_LOTE and P["ep_exp"] == P["ep_base"]:
            r = {**base, "h": base["h"]}
        else:
            r = treinar(f"exp3_lote_{lote}", cfg, texto, tok, device, P["ep_exp"], lote=lote)
        r["nome"] = f"exp3_lote_{lote}"
        salvar_curva(r, curvas)
        recs.append(r)
        rot.append(f"lote={lote}")
        e = P["ep_exp"]
        tl_, vl_ = r["h"]["train_loss"][e], r["h"]["val_loss"][e]
        seg, msp, tks = custo(r)
        linhas.append([lote, r["n_lotes"], r["n_passos"], f3(tl_), f3(vl_), f"{seg:.0f}", f"{msp:.0f}"])
        registrar("exp3", f"lote_{lote}", passos=r["n_passos"], train_loss=tl_, val_loss=vl_,
                  segundos=seg, ms_por_passo=msp, tokens_por_s=tks)
    tabela(f"EXPERIMENTO 3 - Tamanho do lote ({P['ep_exp']} épocas, lr {BASE_LR})",
           ["lote", "lotes/época", "passos", "loss treino", "loss val", "segundos", "ms/passo"], linhas)
    plot_curvas(recs, f"Exp. 3 - Tamanho do lote ({P['ep_exp']} épocas)",
                figuras / "exp3_tamanho_lote.png", rot)

    # ---------------------------------------------------------------- Exp. 4 (dropout)
    print("Exp. 4: dropout...")
    recs, rot, linhas = [], [], []
    for d in P["drops"]:
        if d == 0.0:
            r = {"nome": "exp4_dropout_0", "h": rec["h"], "segundos": rec["segundos"],
                 "n_passos": rec["n_passos"]}               # o treino longo já é dropout 0
        else:
            r = treinar(f"exp4_dropout_{d:g}", {**cfg, "drop_rate": d}, texto, tok, device, P["ep_drop"])
        r["nome"] = f"exp4_dropout_{d:g}"
        salvar_curva(r, curvas)
        recs.append(r)
        rot.append(f"dropout={d:g}")
        e = P["ep_drop"]
        h = r["h"]
        melhor_d = min(range(e + 1), key=lambda i: h["val_loss"][i])
        linhas.append([f"{d:g}", f3(h["train_loss"][e]), f3(h["val_loss"][e]),
                       f3(h["val_loss"][e] - h["train_loss"][e]), f3(h["val_loss"][melhor_d]), melhor_d])
        registrar("exp4", f"dropout_{d:g}", train_loss=h["train_loss"][e], val_loss=h["val_loss"][e],
                  gap=h["val_loss"][e] - h["train_loss"][e], menor_val_loss=h["val_loss"][melhor_d],
                  melhor_epoca_val=melhor_d)
    tabela(f"EXPERIMENTO 4 - Dropout ({P['ep_drop']} épocas, lr {BASE_LR}, lote {BASE_LOTE})",
           ["dropout", "loss treino", "loss val", "val - treino", "menor loss val", "época do menor"], linhas)
    # recorta a curva do treino longo para ep_drop épocas, para comparar nas mesmas épocas
    for r in recs:
        for k in ("passos", "tokens_vistos", "train_loss", "val_loss"):
            r["h"] = {**r["h"], k: r["h"][k][: P["ep_drop"] + 1]}
    plot_curvas(recs, f"Exp. 4 - Dropout ({P['ep_drop']} épocas)", figuras / "exp4_dropout.png", rot)

    # ---------------------------------------------------------------- Exp. 6 (decodificação)
    print("Exp. 6: estratégias de geração...")
    modelos = [("sem treino", novo_modelo(cfg, device)),
               (f"treinado ({P['ep_base']} épocas)", modelo_base),
               (f"sobreajustado ({P['ep_total']} épocas)", modelo_longo)]
    todos, texto_amostras = {}, [f"Prompt: {PROMPT!r} | {P['max_new_tokens']} tokens novos | "
                                 f"seeds {list(P['seeds'])}\n"]
    for nome_m, m in modelos:
        res = comparar_estrategias(m, tok, PROMPT, device, max_new_tokens=P["max_new_tokens"],
                                   seeds=P["seeds"])
        todos[nome_m] = res
        linhas = []
        texto_amostras.append(f"\n{'=' * 70}\nMODELO: {nome_m}\n{'=' * 70}")
        for r in res:
            registrar("exp6", f"{nome_m}|{r['estrategia']}", distinct_1=r["distinct_1"],
                      distinct_2=r["distinct_2"], repeticao_3gram=r["repeticao_3gram"],
                      nll_no_modelo=r["nll_no_modelo"], textos_distintos=r["diversidade_entre_seeds"])
            linhas.append([r["estrategia"], f"{r['distinct_1']:.2f}", f"{r['distinct_2']:.2f}",
                           f"{r['repeticao_3gram']:.2f}", f"{r['diversidade_entre_seeds']:.0%}",
                           f"{r['nll_no_modelo']:.2f}"])
            texto_amostras.append(f"\n--- {r['estrategia']} ---")
            texto_amostras += [f"[seed {s}] {t}" for s, t in zip(P["seeds"], r["textos"])]
        tabela(f"EXPERIMENTO 6 - Decodificação no modelo: {nome_m}",
               ["estratégia", "distinct-1", "distinct-2", "repet. 3-gram", "textos distintos", "NLL no modelo"],
               linhas)
    (saida / "amostras_geracao.txt").write_text("\n".join(texto_amostras), encoding="utf-8")

    estr = [r["estrategia"] for r in todos["sem treino"]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    largura = 0.27
    for j, (nome_m, res) in enumerate(todos.items()):
        xs = [i + (j - 1) * largura for i in range(len(estr))]
        axs[0].bar(xs, [r["distinct_2"] for r in res], largura, label=nome_m)
        axs[1].bar(xs, [r["repeticao_3gram"] for r in res], largura, label=nome_m)
    axs[0].set_title("diversidade (distinct-2; maior = mais variado)")
    axs[1].set_title("repetição de trigramas (maior = mais loops)")
    for ax in axs:
        ax.set_xticks(range(len(estr)))
        ax.set_xticklabels(estr, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Exp. 6 - Estratégias de decodificação x modelo")
    fig.tight_layout()
    fig.savefig(figuras / "exp6_decodificacao.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- saídas
    with open(saida / "metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experimento", "configuracao", "metrica", "valor"])
        w.writerows(linhas_csv)
    (saida / "resumo_metricas.txt").write_text("\n\n".join(resumo) + "\n", encoding="utf-8")
    print(f"Resultados em {saida}")


if __name__ == "__main__":
    main()
