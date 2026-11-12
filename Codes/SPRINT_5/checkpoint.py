"""
Salvamento e carregamento do modelo — Capítulo 5, seção 5.4.

Um modelo PyTorch guarda seus pesos em `model.state_dict()`: um dicionário
{nome da camada: tensor}. Salvar só o state_dict NÃO basta para reconstruir o
modelo depois: é preciso saber a CONFIGURAÇÃO (vocab_size, emb_dim, n_layers...)
para criar um GPTModel com as mesmas formas e então injetar os pesos.

Por isso o checkpoint daqui é um único arquivo .pt com:

    model_state_dict      os pesos aprendidos
    optimizer_state_dict  estado do AdamW (momentos m e v) -> permite RETOMAR o treino
                          exatamente de onde parou (sem ele, o Adam "esquece" o histórico)
    model_config          configuração do GPTModel (para reconstruir a arquitetura)
    train_config          hiperparâmetros do treino (épocas, lr, batch, seed...)
    history               curvas de loss (treino/validação) e amostras de texto
    extra                 loss finais e outras informações livres
    versoes               versão do torch (ajuda a reproduzir)

Além do .pt é gravado um `<nome>_config.json` legível, com a configuração do
modelo e do treino, para identificar o checkpoint sem precisar abri-lo.

Uso (a partir da RAIZ do repositório), carrega e gera texto para provar que
os pesos salvos funcionam:
    python Codes/SPRINT_5/checkpoint.py Experiments/results/sprint5/treino_base/checkpoint.pt
    python Codes/SPRINT_5/checkpoint.py caminho/checkpoint.pt --prompt "Every effort moves you" --temperature 0.8 --top-k 25
"""

import argparse
import json
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[2]
for sprint in ("SPRINT_2", "SPRINT_3", "SPRINT_4", "SPRINT_5"):
    sys.path.insert(0, str(RAIZ / "Codes" / sprint))

from gpt_model import GPTModel  # noqa: E402


def salvar_checkpoint(caminho, model, optimizer, model_config, train_config,
                      history=None, extra=None):
    """Grava checkpoint completo (.pt) + JSON com as configurações."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": dict(model_config),
        "train_config": dict(train_config) if train_config else {},
        "history": history or {},
        "extra": extra or {},
        "versoes": {"torch": str(torch.__version__)},
    }
    torch.save(dados, caminho)
    legivel = {"model_config": dados["model_config"], "train_config": dados["train_config"],
               "extra": dados["extra"], "versoes": dados["versoes"],
               "n_parametros": sum(p.numel() for p in model.parameters()),
               "passos_treinados": (history or {}).get("passos", [0])[-1] if history else 0}
    caminho.with_name(caminho.stem + "_config.json").write_text(
        json.dumps(legivel, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminho


def salvar_so_pesos(caminho, model, model_config):
    """Versão leve (sem otimizador): serve para inferência/compartilhar o modelo."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": dict(model_config),
                "versoes": {"torch": str(torch.__version__)}}, caminho)


def carregar_checkpoint(caminho, device="cpu", carregar_otimizador=False, lr=None, weight_decay=None):
    """
    Reconstrói o modelo a partir do checkpoint.
    Devolve (model, optimizer, meta). model vem em modo eval (inferência).
    optimizer só é criado se carregar_otimizador=True (e houver estado salvo);
    senão é None. meta = dicionário completo do checkpoint (config, history...).
    """
    # weights_only=True: só aceita tensores e tipos simples (mais seguro que pickle livre)
    meta = torch.load(caminho, map_location=device, weights_only=True)
    model = GPTModel(meta["model_config"]).to(device)
    model.load_state_dict(meta["model_state_dict"])
    model.eval()

    optimizer = None
    if carregar_otimizador and meta.get("optimizer_state_dict") is not None:
        tc = meta.get("train_config", {})
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr or tc.get("lr", 5e-4),
                                      weight_decay=weight_decay if weight_decay is not None
                                      else tc.get("weight_decay", 0.1))
        optimizer.load_state_dict(meta["optimizer_state_dict"])
    return model, optimizer, meta


def main():
    ap = argparse.ArgumentParser(description="Carrega um checkpoint e gera texto")
    ap.add_argument("checkpoint")
    ap.add_argument("--prompt", default="Every effort moves you")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    import tiktoken
    from generate import generate, text_to_token_ids, token_ids_to_text

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, meta = carregar_checkpoint(args.checkpoint, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    print("Configuração do modelo:", json.dumps(meta["model_config"]))
    print("Configuração do treino:", json.dumps(meta.get("train_config", {})))
    print("Parâmetros:", f"{sum(p.numel() for p in model.parameters()):,}")

    torch.manual_seed(args.seed)
    ids = text_to_token_ids(args.prompt, tokenizer).to(device)
    out = generate(model, ids, args.max_new_tokens, model.context_length,
                   temperature=args.temperature, top_k=args.top_k)
    print("\nTexto gerado pelo modelo CARREGADO:")
    print(" ", token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    main()
