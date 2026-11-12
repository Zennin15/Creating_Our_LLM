"""
Treinamento do GPT

Cada passo de treino (um lote) faz:

    1. optimizer.zero_grad()   zera os gradientes do passo anterior
    2. loss = cross-entropy    forward: logits -> perda contra os alvos
    3. loss.backward()         retropropagação: calcula d(loss)/d(peso) de todos os pesos
    4. optimizer.step()        AdamW atualiza os pesos usando esses gradientes

Um laço externo repete isso para cada lote do DataLoader (uma ÉPOCA = uma passada
completa pelos dados de treino) e para o número de épocas escolhido.

Durante o treino, a cada `eval_freq` passos medimos a loss de treino e de
validação (com dropout desligado). Treino caindo e validação caindo junto = o
modelo está aprendendo a prever o próximo token. Treino caindo e validação
subindo = SOBREAJUSTE (memorizou os dados de treino).

Como rodar:
    python Codes/SPRINT_5/train.py                       # configuração do livro
    python Codes/SPRINT_5/train.py --epochs 20 --lr 1e-4 --batch-size 4
    python Codes/SPRINT_5/train.py --resume Experiments/results/sprint5/treino_base/checkpoint.pt --epochs 5
Veja todas as opções com --help.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

RAIZ = Path(__file__).resolve().parents[2]
for sprint in ("SPRINT_2", "SPRINT_3", "SPRINT_4", "SPRINT_5"):
    sys.path.insert(0, str(RAIZ / "Codes" / sprint))

from data_loader import GPTDatasetV1  # noqa: E402  (Sprint 2)
from gpt_model import GPTModel, GPT_CONFIG_124M, contar_parametros  # noqa: E402  (Sprint 4)
from generate import generate, text_to_token_ids, token_ids_to_text  # noqa: E402  (Sprint 4)
from loss import calc_loss_batch, avaliar_modelo, perplexity  # noqa: E402
from checkpoint import salvar_checkpoint, carregar_checkpoint  # noqa: E402

# Configuração do livro para o Capítulo 5: GPT-2 small com contexto reduzido
# (256 em vez de 1024) para treinar em CPU; dropout 0.0 (os dados são pequenos).
GPT_CONFIG_TREINO = {**GPT_CONFIG_124M, "context_length": 256, "drop_rate": 0.0}

PROMPT_PADRAO = "Every effort moves you"


def escolher_dispositivo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def criar_loaders(texto, tokenizer, batch_size, max_length, stride=None,
                  train_ratio=0.90, seed=123):
    """
    Divide o TEXTO (não os tokens) em treino/validação, na proporção train_ratio, e
    devolve dois DataLoaders. A divisão é feita no texto, em ordem: validação = final
    do texto, nunca visto no treino. stride=None -> igual a max_length (janelas sem
    sobreposição), como no livro.
    """
    stride = max_length if stride is None else stride
    corte = int(train_ratio * len(texto))
    texto_treino, texto_val = texto[:corte], texto[corte:]

    ds_treino = GPTDatasetV1(texto_treino, tokenizer, max_length, stride)
    ds_val = GPTDatasetV1(texto_val, tokenizer, max_length, stride)
    g = torch.Generator().manual_seed(seed)   # embaralhamento reprodutível
    train_loader = DataLoader(ds_treino, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=g)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def gerar_amostra(model, tokenizer, device, start_context, max_new_tokens=50,
                  temperature=0.0, top_k=None):
    """Gera um texto curto a partir de start_context (greedy por padrão)."""
    ids = text_to_token_ids(start_context, tokenizer).to(device)
    saida = generate(model, ids, max_new_tokens, model.context_length,
                     temperature=temperature, top_k=top_k)
    return token_ids_to_text(saida, tokenizer).replace("\n", " ")


def treinar_modelo(model, train_loader, val_loader, optimizer, device, num_epochs,
                   eval_freq, eval_iter, tokenizer=None, start_context=PROMPT_PADRAO,
                   grad_clip=None, historico=None, verbose=True):
    """
    Laço de treino. Devolve um dicionário `historico` com:
        passos, tokens_vistos, train_loss, val_loss   (uma entrada por avaliação)
        epoca_fim: índice (em `passos`) da última avaliação de cada época
        amostras:  texto gerado ao fim de cada época (se houver tokenizer)
        segundos:  tempo total de treino
    Se `historico` vier de um treino anterior (retomada), continua de onde parou.
    A PRIMEIRA avaliação (passo 0) acontece ANTES de qualquer atualização: é o
    estado inicial do modelo.
    """
    h = historico or {"passos": [], "tokens_vistos": [], "train_loss": [], "val_loss": [],
                      "epoca_fim": [], "amostras": [], "segundos": 0.0}
    passo = h["passos"][-1] if h["passos"] else -1
    tokens_vistos = h["tokens_vistos"][-1] if h["tokens_vistos"] else 0

    def avaliar():
        tl, vl = avaliar_modelo(model, train_loader, val_loader, device, eval_iter)
        h["passos"].append(passo if passo >= 0 else 0)
        h["tokens_vistos"].append(tokens_vistos)
        h["train_loss"].append(tl)
        h["val_loss"].append(vl)
        return tl, vl

    t0 = time.perf_counter()
    if not h["passos"]:                       # estado inicial (antes do treino)
        tl, vl = avaliar()
        if verbose:
            print(f"Estado inicial | loss treino {tl:.3f} (ppl {perplexity(tl):.1f}) | "
                  f"loss val {vl:.3f} (ppl {perplexity(vl):.1f})")
    passo = max(passo, 0)

    for epoca in range(num_epochs):
        model.train()                         # liga dropout
        for x, y in train_loader:
            optimizer.zero_grad()                                  # 1
            loss = calc_loss_batch(x, y, model, device)            # 2
            loss.backward()                                        # 3
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()                                       # 4
            passo += 1
            tokens_vistos += x.numel()

            if passo % eval_freq == 0:
                tl, vl = avaliar()
                if verbose:
                    print(f"Época {epoca + 1:>3} (passo {passo:>5}) | "
                          f"loss treino {tl:.3f} | loss val {vl:.3f}")

        if h["passos"][-1] != passo:          # garante uma avaliação no fim da época
            avaliar()
        h["epoca_fim"].append(len(h["passos"]) - 1)
        if tokenizer is not None:
            amostra = gerar_amostra(model, tokenizer, device, start_context)
            h["amostras"].append(amostra)
            if verbose:
                print(f"   amostra: {amostra}")

    h["segundos"] += time.perf_counter() - t0
    return h


def novo_otimizador(model, lr, weight_decay):
    """AdamW: Adam com weight decay desacoplado (regularização L2 correta)."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def salvar_historico_csv(h, caminho):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["passo", "tokens_vistos", "train_loss", "val_loss",
                    "train_ppl", "val_ppl"])
        for p, t, tl, vl in zip(h["passos"], h["tokens_vistos"], h["train_loss"], h["val_loss"]):
            w.writerow([p, t, f"{tl:.5f}", f"{vl:.5f}",
                        f"{perplexity(tl):.3f}", f"{perplexity(vl):.3f}"])


def plotar_perdas(h, caminho, titulo="Loss de treino e validação"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["passos"], h["train_loss"], label="treino")
    ax.plot(h["passos"], h["val_loss"], label="validação", linestyle="--")
    ax.set_xlabel("passos de treino")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Treina o GPT no the-verdict.txt")
    ap.add_argument("--dados", default=str(RAIZ / "the-verdict.txt"))
    ap.add_argument("--saida", default=str(RAIZ / "Experiments" / "results" / "sprint5" / "treino_base"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--eval-iter", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--context-length", type=int, default=GPT_CONFIG_TREINO["context_length"])
    ap.add_argument("--emb-dim", type=int, default=GPT_CONFIG_TREINO["emb_dim"])
    ap.add_argument("--n-heads", type=int, default=GPT_CONFIG_TREINO["n_heads"])
    ap.add_argument("--n-layers", type=int, default=GPT_CONFIG_TREINO["n_layers"])
    ap.add_argument("--drop-rate", type=float, default=GPT_CONFIG_TREINO["drop_rate"])
    ap.add_argument("--prompt", default=PROMPT_PADRAO)
    ap.add_argument("--resume", default=None, help="checkpoint .pt para continuar o treino")
    args = ap.parse_args()

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    device = escolher_dispositivo()
    torch.manual_seed(args.seed)

    texto = Path(args.dados).read_text(encoding="utf-8")
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model, optimizer, meta = carregar_checkpoint(args.resume, device, carregar_otimizador=True)
        cfg_modelo, historico = meta["model_config"], meta["history"]
        cfg_treino = meta["train_config"]
        cfg_treino["epochs_total"] = cfg_treino.get("epochs_total", cfg_treino["epochs"]) + args.epochs
        print(f"Retomando de {args.resume}: {historico['passos'][-1]} passos já feitos.")
    else:
        cfg_modelo = {**GPT_CONFIG_TREINO, "context_length": args.context_length,
                      "emb_dim": args.emb_dim, "n_heads": args.n_heads,
                      "n_layers": args.n_layers, "drop_rate": args.drop_rate}
        model = GPTModel(cfg_modelo).to(device)
        optimizer = novo_otimizador(model, args.lr, args.weight_decay)
        historico = None
        cfg_treino = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                      "weight_decay": args.weight_decay, "optimizer": "AdamW",
                      "grad_clip": args.grad_clip, "eval_freq": args.eval_freq,
                      "eval_iter": args.eval_iter, "train_ratio": args.train_ratio,
                      "seed": args.seed, "dados": Path(args.dados).name}

    train_loader, val_loader = criar_loaders(
        texto, tokenizer, cfg_treino["batch_size"], cfg_modelo["context_length"],
        train_ratio=cfg_treino["train_ratio"], seed=cfg_treino["seed"])

    print(f"Dispositivo: {device} | parâmetros: {contar_parametros(model)['total']:,}")
    print(f"Lotes de treino: {len(train_loader)} | lotes de validação: {len(val_loader)}")
    print(f"Modelo: {json.dumps(cfg_modelo)}")
    print(f"Treino: {json.dumps(cfg_treino)}\n")

    historico = treinar_modelo(
        model, train_loader, val_loader, optimizer, device, args.epochs,
        cfg_treino["eval_freq"], cfg_treino["eval_iter"], tokenizer, args.prompt,
        grad_clip=cfg_treino.get("grad_clip"), historico=historico)

    # Loss final com TODOS os lotes (sem estimativa por eval_iter)
    tl, vl = avaliar_modelo(model, train_loader, val_loader, device, None)
    print(f"\nFinal | loss treino {tl:.3f} (ppl {perplexity(tl):.2f}) | "
          f"loss val {vl:.3f} (ppl {perplexity(vl):.2f}) | {historico['segundos']:.0f} s")

    salvar_checkpoint(saida / "checkpoint.pt", model, optimizer, cfg_modelo, cfg_treino,
                      historico, extra={"loss_final_treino": tl, "loss_final_val": vl})
    salvar_historico_csv(historico, saida / "perdas.csv")
    plotar_perdas(historico, saida / "perdas.png")
    (saida / "amostras.txt").write_text(
        "\n".join(f"Época {i + 1}: {a}" for i, a in enumerate(historico["amostras"])),
        encoding="utf-8")
    print(f"Arquivos salvos em {saida}")


if __name__ == "__main__":
    main()
