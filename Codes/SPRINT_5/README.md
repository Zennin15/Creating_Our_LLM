# Sprint 5 — Treinamento e avaliação do GPT (Capítulo 5)

Treina o GPT da Sprint 4 para prever o próximo token, mede o aprendizado com a loss
(cross-entropy) e a perplexity, compara estratégias de geração de texto e salva/carrega
os pesos.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `loss.py` | Cross-entropy (manual e PyTorch), loss de um lote / de um DataLoader, `avaliar_modelo` (treino e validação), perplexity |
| `train.py` | Divisão treino/validação, laço de treino (épocas, lotes, `zero_grad` → loss → `backward` → AdamW), registro das perdas, retomada de treino (`--resume`) |
| `checkpoint.py` | `salvar_checkpoint` / `carregar_checkpoint` (pesos + otimizador + configuração + histórico) e demonstração de inferência com o modelo carregado |
| `decoding_strategies.py` | Compara greedy, amostragem, temperatura e top-k no mesmo prompt, com métricas de diversidade e repetição |
| `test_training.py` | 19 testes de sanidade (não precisam de internet nem tokenizer) |
| `../../Experiments/experiments_sprint5.py` | Experimentos 0 a 6 (estado inicial, épocas, lr, lote, dropout, custo, decodificação) |

Reaproveita `GPTDatasetV1` (Sprint 2), `GPTModel` (Sprint 4) e `generate` com temperatura e
top-k (Sprint 4).

## Como rodar (a partir da RAIZ do repositório)

```
pip install torch tiktoken matplotlib
python Codes/SPRINT_5/test_training.py          # testes

python Codes/SPRINT_5/loss.py                   # demonstração da cross-entropy / perplexity
python Codes/SPRINT_5/train.py                  # treino com os parâmetros do livro
python Codes/SPRINT_5/train.py --epochs 20 --lr 1e-4 --batch-size 4   # outra configuração
python Codes/SPRINT_5/train.py --resume Experiments/results/sprint5/treino_base/checkpoint.pt --epochs 5

python Codes/SPRINT_5/checkpoint.py Experiments/results/sprint5/treino_base/checkpoint.pt
python Codes/SPRINT_5/decoding_strategies.py Experiments/results/sprint5/treino_base/checkpoint.pt

python Experiments/experiments_sprint5.py --preset rapido   # experimentos (menor custo)
python Experiments/experiments_sprint5.py                   # experimentos completos
```

O `train.py` grava em `Experiments/results/sprint5/treino_base/`: `checkpoint.pt`,
`checkpoint_config.json` (configuração legível), `perdas.csv`, `perdas.png` e `amostras.txt`.

## Parâmetros do treino-base (configuração do livro)

| Parâmetro | Valor |
|---|---|
| Modelo | GPT-2 small (12 camadas, 12 heads, emb_dim 768, vocab 50257), `context_length` 256, dropout 0.0 |
| Otimizador | AdamW, `lr = 5e-4`, `weight_decay = 0.1` |
| Épocas / lote | 10 / 2 |
| Dados | `the-verdict.txt`, 90% treino / 10% validação (divisão no texto, em ordem) |
| Avaliação | a cada 5 passos, com 5 lotes (`--eval-freq`, `--eval-iter`); loss final com todos os lotes |
| Seed | 123 |

Os experimentos usam um modelo menor (emb_dim 256, 4 blocos, 8 heads, contexto 256) para
rodar em CPU em tempo razoável; a configuração está no topo de `experiments_sprint5.py`.

## Observações

- **Pesos não vão para o Git**: o `.gitignore` ignora `*.pt` (o GPT-2 small tem ~650 MB só nos
  pesos, e o GitHub recusa arquivos acima de 100 MB). O `checkpoint_config.json` e os CSVs
  vão no repositório; o `.pt` se regenera com o comando de treino (seed fixa).
- A loss de treino medida com `--eval-iter` usa lotes sorteados (o DataLoader de treino
  embaralha), então varia um pouco entre avaliações. Nos experimentos usamos todos os lotes.
- O `--preset smoke` dos experimentos usa um tokenizer de bytes e um modelo minúsculo: só
  confere que o pipeline roda; seus números não servem para o relatório.
- Os tempos dependem da máquina; compare tendências, não valores absolutos.
