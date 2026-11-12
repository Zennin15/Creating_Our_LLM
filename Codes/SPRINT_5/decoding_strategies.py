"""
Estratégias de decodificação — Capítulo 5, seção 5.3.

O modelo só entrega LOGITS. Quem decide qual token vira texto é a ESTRATÉGIA DE
DECODIFICAÇÃO (a função `generate`, Sprint 4). Aqui comparamos, a partir do MESMO
contexto inicial:

    greedy            temperature=0           sempre o token mais provável
    amostragem pura   temperature=1           sorteio com as probabilidades do modelo
    temperatura       temperature<1 ou >1     <1 afia a distribuição, >1 achata
    top-k             top_k=k                 sorteia só entre os k tokens mais prováveis

IMPORTANTE para a análise: a decodificação NÃO muda os pesos. Só muda como lemos
as probabilidades que o modelo já calculou. Por isso comparamos as mesmas
estratégias em modelos de qualidade diferente (sem treino x treinado x
sobreajustado): diferenças ENTRE estratégias, no mesmo modelo, vêm da decodificação;
diferenças entre modelos, com a mesma estratégia, vêm do treinamento dos pesos.

Métricas automáticas (apoiam a leitura, não substituem ler os textos):
    distinct_1 / distinct_2   fração de tokens / bigramas ÚNICOS no texto gerado.
                              Alto = diverso; baixo = repetitivo.
    repeticao_3gram           fração de trigramas que já tinham aparecido antes.
                              Alto = o texto entrou em loop.
    diversidade_entre_seeds   fração de textos DISTINTOS entre várias seeds
                              (greedy dá sempre 1 texto: valor 1/n_seeds).
    nll_no_modelo             loss média do próprio modelo sobre o texto que ele
                              gerou. Baixa = o texto é "fácil" para o modelo (o greedy
                              tira notas baixas, inclusive quando está em loop). Não
                              mede coerência; só mostra o quanto o texto segue a
                              distribuição do modelo.
Coerência (o texto faz sentido?) só se avalia lendo — veja amostras_geracao.txt.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[2]
for sprint in ("SPRINT_4",):
    sys.path.insert(0, str(RAIZ / "Codes" / sprint))

from generate import generate, text_to_token_ids, token_ids_to_text  # noqa: E402

# (rótulo, temperature, top_k)
ESTRATEGIAS_PADRAO = [
    ("greedy",                0.0, None),
    ("amostragem T=1.0",      1.0, None),
    ("T=0.7",                 0.7, None),
    ("T=1.5",                 1.5, None),
    ("top-k=5, T=1.0",        1.0, 5),
    ("top-k=50, T=1.0",       1.0, 50),
    ("top-k=5, T=0.7",        0.7, 5),
]


def distinct_n(tokens, n):
    """Fração de n-gramas únicos (1.0 = nenhum repetido)."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def repeticao_ngram(tokens, n=3):
    """Fração de n-gramas que já haviam aparecido antes no próprio texto."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return float("nan")
    vistos, rep = set(), 0
    for g in grams:
        rep += g in vistos
        vistos.add(g)
    return rep / len(grams)


@torch.no_grad()
def nll_da_continuacao(model, ids_completos, n_prompt, device):
    """Loss média do modelo sobre os tokens GERADOS (os depois do prompt)."""
    model.eval()
    ids = ids_completos[:, -model.context_length:]
    n_prompt = n_prompt - (ids_completos.shape[1] - ids.shape[1])
    logits = model(ids[:, :-1].to(device))                      # prevê ids[:, 1:]
    alvo = ids[:, 1:].to(device)
    perdas = F.cross_entropy(logits.flatten(0, 1), alvo.flatten(), reduction="none")
    perdas = perdas.view(alvo.shape)[:, max(n_prompt - 1, 0):]   # só a parte gerada
    return perdas.mean().item()


def comparar_estrategias(model, tokenizer, prompt, device, estrategias=None,
                         max_new_tokens=50, seeds=(1, 2, 3, 4, 5)):
    """
    Para cada estratégia gera um texto por seed, a partir do MESMO prompt.
    Devolve lista de dicts: estrategia, temperature, top_k, textos (um por seed) e as
    métricas médias sobre as seeds.
    """
    estrategias = estrategias or ESTRATEGIAS_PADRAO
    ids0 = text_to_token_ids(prompt, tokenizer).to(device)
    n_prompt = ids0.shape[1]
    linhas = []
    for rotulo, temp, k in estrategias:
        textos, d1, d2, rep, nll = [], [], [], [], []
        for s in seeds:
            torch.manual_seed(s)
            out = generate(model, ids0, max_new_tokens, model.context_length,
                           temperature=temp, top_k=k)
            novos = out[0, n_prompt:].tolist()
            textos.append(token_ids_to_text(out, tokenizer).replace("\n", " "))
            d1.append(distinct_n(novos, 1))
            d2.append(distinct_n(novos, 2))
            rep.append(repeticao_ngram(novos, 3))
            nll.append(nll_da_continuacao(model, out.cpu(), n_prompt, device))
        media = lambda v: sum(v) / len(v)
        linhas.append({
            "estrategia": rotulo, "temperature": temp, "top_k": k, "textos": textos,
            "distinct_1": media(d1), "distinct_2": media(d2), "repeticao_3gram": media(rep),
            "nll_no_modelo": media(nll),
            "diversidade_entre_seeds": len(set(textos)) / len(textos),
        })
    return linhas


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compara greedy x amostragem x temperatura x top-k")
    ap.add_argument("checkpoint")
    ap.add_argument("--prompt", default="Every effort moves you")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    args = ap.parse_args()

    import tiktoken
    sys.path.insert(0, str(RAIZ / "Codes" / "SPRINT_5"))
    from checkpoint import carregar_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = carregar_checkpoint(args.checkpoint, device)
    tok = tiktoken.get_encoding("gpt2")
    for r in comparar_estrategias(model, tok, args.prompt, device,
                                  max_new_tokens=args.max_new_tokens):
        print(f"\n== {r['estrategia']} | distinct-2 {r['distinct_2']:.2f} | "
              f"rep.3-gram {r['repeticao_3gram']:.2f} | textos distintos "
              f"{r['diversidade_entre_seeds']:.0%}")
        print("  seed 1:", r["textos"][0])
