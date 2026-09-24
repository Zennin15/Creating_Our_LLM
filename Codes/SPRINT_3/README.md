# Sprint 3 — Mecanismo de Attention (Capítulo 3)

Implementação Self-Attention>Scaled Dot-Product>Causal Attention>Multi-Head Attention.
O `MultiHeadAttention` será usado nos blocos Transformer da Sprint 4.

## Como rodar (a partir da RAIZ do repositório)


pip install torch tiktoken matplotlib
python Codes/SPRINT_3/test_attention.py         # testes
python Codes/SPRINT_3/causal_attention.py       # demos individuais
python Experiments/experiments_sprint3.py       # experimentos + figuras + métricas


## Como ler uma matriz de atenção
Linha = token que "pergunta" (query); coluna = token "consultado" (key); cada linha soma 1.
Na atenção causal, tudo acima da diagonal é 0.
