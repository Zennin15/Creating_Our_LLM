# Sprint 4 — Arquitetura GPT e geração de texto (Capítulo 4)

Integra a `MultiHeadAttention` da Sprint 3 aos demais componentes do Transformer
e constrói um GPT simplificado capaz de gerar texto token a token.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `layer_norm.py` | LayerNorm implementada do zero (média 0, variância 1 por token, + scale/shift) |
| `gelu.py` | GELU (aproximação por tanh, a do GPT-2) |
| `feed_forward.py` | FeedForward: emb_dim -> 4*emb_dim -> emb_dim, com GELU |
| `transformer_block.py` | Transformer Block: pre-LayerNorm + MultiHeadAttention (Sprint 3) + FeedForward + 2 conexões residuais |
| `gpt_model.py` | `GPTModel`: embeddings (Sprint 2) -> N blocos -> LayerNorm final -> Language Model Head -> logits |
| `generate.py` | Geração autoregressiva: greedy, temperatura, top-k; helpers de texto <-> Token IDs |
| `test_gpt.py` | 19 testes de sanidade (não precisam de internet nem tokenizer) |

## Como rodar (a partir da RAIZ do repositório)

```
pip install torch tiktoken matplotlib
python Codes/SPRINT_4/test_gpt.py              # testes
python Codes/SPRINT_4/gpt_model.py             # formatos e contagem de parâmetros (124M)
python Codes/SPRINT_4/generate.py              # um passo de geração + texto gerado
python Experiments/experiments_sprint4.py      # experimentos + figuras + métricas
```

## Fluxo dos dados (B = batch, T = nº de tokens)

```
Token IDs (B,T) -> tok_emb + pos_emb (B,T,emb_dim) -> dropout
  -> N x TransformerBlock (B,T,emb_dim) -> LayerNorm final (B,T,emb_dim)
  -> out_head (Linear, sem bias) -> logits (B,T,vocab_size)
```

## Observações

- O modelo **não é treinado** (treino é o Capítulo 5): o texto gerado é sem sentido, e isso é esperado.
- O `out_head` tem pesos próprios (sem weight tying): 163.009.536 parâmetros no GPT-2 small.
  Com weight tying seriam 124.412.160 (o "124M" do GPT-2).
- Os tempos dos experimentos dependem da máquina; compare tendências, não valores absolutos.
