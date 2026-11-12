"""
Testes de sanidade da Sprint 5 (loss, perplexity, treino, checkpoint, decodificação).
Não precisam de internet nem do tiktoken: usam um tokenizer de bytes (vocabulário de
256 tokens) e um modelo minúsculo.

"""

import math
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[2]
for sprint in ("SPRINT_2", "SPRINT_3", "SPRINT_4", "SPRINT_5"):
    sys.path.insert(0, str(RAIZ / "Codes" / sprint))

from gpt_model import GPTModel  # noqa: E402
from generate import generate  # noqa: E402
from loss import (cross_entropy_manual, calc_loss_batch, calc_loss_loader,  # noqa: E402
                  avaliar_modelo, perplexity, loss_inicial_esperada)
from train import criar_loaders, treinar_modelo, novo_otimizador  # noqa: E402
from checkpoint import salvar_checkpoint, carregar_checkpoint, salvar_so_pesos  # noqa: E402
from decoding_strategies import (distinct_n, repeticao_ngram,  # noqa: E402
                                 comparar_estrategias)


class ByteTokenizer:
    """Tokenizer de teste: cada byte UTF-8 é um token (vocabulário = 256)."""
    n_vocab = 256

    def encode(self, texto, allowed_special=None):
        return list(texto.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


TOK = ByteTokenizer()
CFG = {"vocab_size": 256, "context_length": 16, "emb_dim": 32, "n_heads": 4,
       "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False}
TEXTO = ("the quick brown fox jumps over the lazy dog. " * 40)   # repetitivo: dá para memorizar
DEV = torch.device("cpu")


def novo_modelo(seed=123, **mud):
    torch.manual_seed(seed)
    return GPTModel({**CFG, **mud})


def loaders(batch=4):
    return criar_loaders(TEXTO, TOK, batch_size=batch, max_length=CFG["context_length"],
                         train_ratio=0.9, seed=1)


# ---------------------------------------------------------------- loss / perplexity
def teste_cross_entropy_manual_igual_pytorch():
    torch.manual_seed(0)
    logits, alvos = torch.randn(20, 50), torch.randint(0, 50, (20,))
    assert torch.allclose(cross_entropy_manual(logits, alvos), F.cross_entropy(logits, alvos), atol=1e-6)


def teste_logits_uniformes_dao_log_v():
    V = 100
    loss = F.cross_entropy(torch.zeros(7, V), torch.randint(0, V, (7,)))
    assert abs(loss.item() - loss_inicial_esperada(V)) < 1e-5
    assert abs(perplexity(loss.item()) - V) < 1e-3          # perplexity = V


def teste_certeza_no_alvo_da_loss_zero_e_ppl_1():
    logits = torch.full((3, 10), -50.0)
    alvos = torch.tensor([2, 5, 7])
    logits[torch.arange(3), alvos] = 50.0
    loss = F.cross_entropy(logits, alvos).item()
    assert loss < 1e-6 and abs(perplexity(loss) - 1.0) < 1e-5


def teste_perplexity_nao_estoura():
    assert math.isfinite(perplexity(1e6)) and math.isnan(perplexity(float("nan")))


def teste_loss_inicial_perto_de_log_v():
    train_loader, val_loader = loaders()
    model = novo_modelo()
    tl, vl = avaliar_modelo(model, train_loader, val_loader, DEV)
    esperado = loss_inicial_esperada(CFG["vocab_size"])
    assert esperado - 0.5 < tl < esperado + 2.0, (tl, esperado)
    assert esperado - 0.5 < vl < esperado + 2.0, (vl, esperado)


def teste_loss_loader_e_media_dos_lotes():
    from torch.utils.data import DataLoader
    # loader SEM embaralhar: com shuffle=True cada passada pega lotes em ordem diferente
    train_loader = DataLoader(loaders()[0].dataset, batch_size=4, shuffle=False)
    model = novo_modelo().eval()
    with torch.no_grad():
        por_lote = [calc_loss_batch(x, y, model, DEV).item() for x, y in train_loader]
        assert abs(calc_loss_loader(train_loader, model, DEV) - sum(por_lote) / len(por_lote)) < 1e-6
        assert abs(calc_loss_loader(train_loader, model, DEV, num_batches=2) - sum(por_lote[:2]) / 2) < 1e-6


def teste_avaliar_restaura_modo_e_nao_muda_pesos():
    train_loader, val_loader = loaders()
    model = novo_modelo().train()
    antes = [p.clone() for p in model.parameters()]
    avaliar_modelo(model, train_loader, val_loader, DEV)
    assert model.training
    assert all(torch.equal(a, p) for a, p in zip(antes, model.parameters()))
    model.eval()
    avaliar_modelo(model, train_loader, val_loader, DEV)
    assert not model.training


# ---------------------------------------------------------------- dados
def teste_alvo_e_entrada_deslocada_em_1():
    train_loader, _ = loaders(batch=2)
    x, y = next(iter(train_loader))
    assert torch.equal(x[:, 1:], y[:, :-1])


def teste_split_treino_validacao_sem_vazamento():
    train_loader, val_loader = loaders()
    ds_t, ds_v = train_loader.dataset, val_loader.dataset
    corte = int(0.9 * len(TEXTO))
    assert len(ds_t) > 0 and len(ds_v) > 0
    # a validação vem do FINAL do texto: o 1º trecho dela é prefixo do texto[corte:]
    assert TOK.decode(ds_v.input_ids[0].tolist()) == TEXTO[corte:corte + CFG["context_length"]]
    # e o último trecho de treino termina antes do corte
    n_tok_treino = len(ds_t) * CFG["context_length"]
    assert n_tok_treino <= corte


# ---------------------------------------------------------------- treino
def teste_gradientes_existem_e_pesos_mudam_apos_step():
    train_loader, _ = loaders()
    model = novo_modelo().train()
    opt = novo_otimizador(model, 1e-3, 0.1)
    antes = {n: p.detach().clone() for n, p in model.named_parameters()}
    x, y = next(iter(train_loader))
    opt.zero_grad()
    calc_loss_batch(x, y, model, DEV).backward()
    assert all(p.grad is not None for p in model.parameters())
    opt.step()
    mudou = sum(not torch.equal(antes[n], p) for n, p in model.named_parameters())
    assert mudou >= len(antes) // 2


def teste_zero_grad_limpa_gradientes():
    model = novo_modelo()
    opt = novo_otimizador(model, 1e-3, 0.1)
    x, y = next(iter(loaders()[0]))
    calc_loss_batch(x, y, model, DEV).backward()
    opt.zero_grad()      # PyTorch recente: set_to_none=True
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.parameters())


def teste_treino_reduz_loss_de_treino_e_validacao():
    train_loader, val_loader = loaders()
    model = novo_modelo()
    opt = novo_otimizador(model, 3e-3, 0.0)
    h = treinar_modelo(model, train_loader, val_loader, opt, DEV, num_epochs=8,
                       eval_freq=2, eval_iter=None, tokenizer=TOK, start_context="the quick",
                       verbose=False)
    assert h["passos"][0] == 0                                   # estado inicial registrado
    assert h["train_loss"][-1] < h["train_loss"][0] - 1.0, h["train_loss"]
    assert h["val_loss"][-1] < h["val_loss"][0] - 1.0, h["val_loss"]
    assert len(h["epoca_fim"]) == 8 and len(h["amostras"]) == 8
    assert len(h["passos"]) == len(h["train_loss"]) == len(h["val_loss"]) == len(h["tokens_vistos"])
    assert h["tokens_vistos"][-1] == 8 * len(train_loader) * 4 * CFG["context_length"]


def teste_treino_e_reprodutivel_com_mesma_seed():
    def roda():
        tl, vl = loaders()
        m = novo_modelo()
        h = treinar_modelo(m, tl, vl, novo_otimizador(m, 1e-3, 0.1), DEV, 2, 2, None,
                           verbose=False)
        return h["train_loss"]
    assert roda() == roda()


# ---------------------------------------------------------------- checkpoint
def teste_checkpoint_ida_e_volta_mesmos_logits_e_texto():
    model = novo_modelo()
    opt = novo_otimizador(model, 1e-3, 0.1)
    x, y = next(iter(loaders()[0]))
    calc_loss_batch(x, y, model, DEV).backward()
    opt.step()
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        cam = Path(d) / "ck.pt"
        salvar_checkpoint(cam, model, opt, CFG, {"lr": 1e-3, "weight_decay": 0.1, "epochs": 1},
                          history={"passos": [0, 1]})
        assert (Path(d) / "ck_config.json").exists()
        novo, _, meta = carregar_checkpoint(cam, DEV)
    assert meta["model_config"] == CFG
    assert not novo.training                                        # vem em modo de inferência
    assert torch.allclose(model(x), novo(x), atol=1e-6)
    prompt = torch.tensor([[116, 104, 101]])
    a = generate(model, prompt, 10, CFG["context_length"])
    b = generate(novo, prompt, 10, CFG["context_length"])
    assert torch.equal(a, b)


def teste_checkpoint_restaura_otimizador_e_retoma_treino():
    train_loader, val_loader = loaders()
    model = novo_modelo()
    opt = novo_otimizador(model, 3e-3, 0.0)
    h = treinar_modelo(model, train_loader, val_loader, opt, DEV, 2, 2, None, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        cam = Path(d) / "ck.pt"
        salvar_checkpoint(cam, model, opt, CFG, {"lr": 3e-3, "weight_decay": 0.0}, history=h)
        m2, o2, meta = carregar_checkpoint(cam, DEV, carregar_otimizador=True)
    # estado do Adam (passo e momentos) foi restaurado
    assert len(o2.state) == len(opt.state) > 0
    p0 = next(iter(opt.param_groups[0]["params"]))
    q0 = next(iter(o2.param_groups[0]["params"]))
    assert torch.allclose(opt.state[p0]["exp_avg"], o2.state[q0]["exp_avg"])
    # continua a contagem de passos do histórico
    ultimo = meta["history"]["passos"][-1]
    h2 = treinar_modelo(m2, train_loader, val_loader, o2, DEV, 1, 2, None,
                        historico=meta["history"], verbose=False)
    assert h2["passos"][-1] == ultimo + len(train_loader)
    assert h2["val_loss"][0] == h["val_loss"][0]                    # estado inicial preservado


def teste_so_pesos_carrega_para_inferencia():
    model = novo_modelo().eval()
    with tempfile.TemporaryDirectory() as d:
        cam = Path(d) / "pesos.pt"
        salvar_so_pesos(cam, model, CFG)
        novo, opt, _ = carregar_checkpoint(cam, DEV, carregar_otimizador=True)
    assert opt is None
    x = torch.randint(0, 256, (1, 8))
    assert torch.allclose(model(x), novo(x), atol=1e-6)


# ---------------------------------------------------------------- decodificação
def teste_metricas_de_diversidade_e_repeticao():
    assert distinct_n([1, 2, 3, 4], 1) == 1.0
    assert distinct_n([1, 1, 1, 1], 1) == 0.25
    assert repeticao_ngram([1, 2, 3, 4, 5], 3) == 0.0
    assert repeticao_ngram([1, 2, 1, 2, 1, 2, 1, 2], 3) > 0.5


def teste_greedy_deterministico_e_topk1_igual_greedy():
    model = novo_modelo().eval()
    p = torch.tensor([[1, 2, 3]])
    g1 = generate(model, p, 8, 16, temperature=0.0)
    g2 = generate(model, p, 8, 16, temperature=0.0)
    assert torch.equal(g1, g2)
    torch.manual_seed(7)
    k1 = generate(model, p, 8, 16, temperature=1.0, top_k=1)
    assert torch.equal(g1, k1)                                      # top-1 = greedy


def teste_amostragem_varia_com_seed_e_comparar_estrategias_roda():
    model = novo_modelo().eval()
    res = comparar_estrategias(model, TOK, "the quick", DEV, max_new_tokens=12, seeds=(1, 2, 3, 4))
    por = {r["estrategia"]: r for r in res}
    assert por["greedy"]["diversidade_entre_seeds"] == 0.25         # 1 texto só para 4 seeds
    assert por["amostragem T=1.0"]["diversidade_entre_seeds"] > 0.25
    assert all(len(r["textos"]) == 4 for r in res)
    assert all(math.isfinite(r["nll_no_modelo"]) for r in res)


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    falhas = 0
    for nome, fn in testes:
        try:
            fn()
            print(f"  OK   {nome}")
        except Exception as e:  # noqa: BLE001
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    sys.exit(1 if falhas else 0)
