"""
Inferência e geração autoregressiva de texto — Capítulo 4 (seção 4.7) e
extensões do Capítulo 5 (temperatura e top-k).

Ideia central: o modelo só sabe prever UM token por vez. Para gerar um texto:

    repita N vezes:
      1. pegue os últimos `context_size` tokens da sequência atual
      2. rode o modelo -> logits (B, T, vocab_size)
      3. fique só com os logits da ÚLTIMA posição -> (B, vocab_size)
      4. (opcional) top-k: zere (-inf) tudo que não está entre os k maiores logits
      5. escolha o próximo token:
            temperature = 0  -> greedy: argmax (sempre o mais provável)
            temperature > 0  -> softmax(logits / temperature) e SORTEIO (multinomial)
      6. anexe o token escolhido à sequência  <- é isso que torna "autoregressivo"

Relação logits -> probabilidades -> token:
    logits  : pontuações brutas (qualquer valor real)
    softmax : transforma em probabilidades (positivas, somam 1)
    seleção : argmax (determinístico) ou sorteio (estocástico)
O argmax dos logits e o argmax das probabilidades dão o MESMO token (o softmax
é monotônico); o softmax só é necessário quando queremos sortear.

Temperatura: divide os logits antes do softmax.
    < 1  -> distribuição mais "afiada" (mais previsível)
    = 1  -> distribuição original do modelo
    > 1  -> distribuição mais "achatada" (mais aleatório)
Top-k: restringe o sorteio aos k tokens mais prováveis (corta a "cauda" de
tokens improváveis que causam texto sem sentido).

IMPORTANTE: modelo SEM treino (pesos aleatórios) gera texto sem sentido — isso
é esperado. O treinamento é assunto do próximo capítulo e da proxima SPRINT.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """str -> tensor (1, T) com Token IDs (o 1 é a dimensão de batch)."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """tensor (1, T) -> str."""
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    idx: (B, T) Token IDs iniciais (o "prompt").
    Retorna (B, T + max_new_tokens) — ou menos, se eos_id for gerado.

    temperature=0.0 -> greedy (determinístico). top_k=None -> sem corte.
    """
    # Inferência: desliga o dropout (senão o resultado seria aleatório).
    estava_treinando = model.training
    model.eval()

    for _ in range(max_new_tokens):
        # 1. Se a sequência passar do contexto, usamos só os últimos tokens
        idx_cond = idx[:, -context_size:]

        # 2-3. Logits da última posição: (B, vocab_size)
        logits = model(idx_cond)[:, -1, :]

        # 4. Top-k
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]            # k-ésimo maior logit
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # 5. Seleção do próximo token
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        # 6. Anexa e repete
        idx = torch.cat((idx, idx_next), dim=1)

    model.train(estava_treinando)
    return idx


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Versão greedy (do livro): sempre escolhe o token de maior logit."""
    return generate(model, idx, max_new_tokens, context_size, temperature=0.0)


@torch.no_grad()
def passo_a_passo(model, idx, tokenizer, top_n=5):
    """Mostra UM passo de geração: logits -> probabilidades -> próximo token."""
    estava_treinando = model.training
    model.eval()
    logits = model(idx)                          # (1, T, vocab_size)
    ultimos = logits[:, -1, :]                   # (1, vocab_size)
    probs = torch.softmax(ultimos, dim=-1)       # (1, vocab_size)
    top_p, top_i = torch.topk(probs[0], top_n)

    print(f"  logits (todas as posições): {tuple(logits.shape)}")
    print(f"  logits da última posição  : {tuple(ultimos.shape)}")
    print(f"  soma das probabilidades   : {probs.sum().item():.4f}")
    print(f"  Top-{top_n} tokens mais prováveis:")
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        print(f"    id {i:>6}  p={p:.5f}  {tokenizer.decode([i])!r}")
    model.train(estava_treinando)


if __name__ == "__main__":
    import tiktoken
    from gpt_model import GPTModel, GPT_CONFIG_124M

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)

    prompt = "Hello, I am"
    idx = text_to_token_ids(prompt, tokenizer)
    print(f"Prompt: {prompt!r} -> Token IDs {idx.tolist()} (shape {tuple(idx.shape)})")

    print("\nUm passo de geração:")
    passo_a_passo(model, idx, tokenizer)

    out = generate_text_simple(model, idx, max_new_tokens=6,
                               context_size=GPT_CONFIG_124M["context_length"])
    print("\nGreedy (sem treino -> texto sem sentido, é esperado):")
    print(" ", token_ids_to_text(out, tokenizer))
