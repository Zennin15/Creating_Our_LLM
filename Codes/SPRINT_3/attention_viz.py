"""
Visualização das matrizes de atenção (heatmaps).

Como ler uma matriz de atenção (tokens x tokens):
  - LINHA  = token que "pergunta" (query): quem está atendendo.
  - COLUNA = token "consultado" (key): quem está sendo atendido.
  - Cada célula = attention weight (0 a 1). Cada LINHA soma 1.
  - Na atenção causal, tudo ACIMA da diagonal é 0 (o futuro é bloqueado).

"""

import os

import matplotlib.pyplot as plt


def _to_numpy(w):
    return w.detach().cpu().numpy()


def plot_attention_matrix(weights, tokens, title="Matriz de atenção", ax=None,
                          annotate=True, cmap="viridis"):
    """weights: tensor (T, T). tokens: lista de T strings.
    Se `ax` for None, cria uma figura nova e a devolve; senão desenha em `ax`."""
    w = _to_numpy(weights)
    n = len(tokens)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.75 * n + 2.5))

    im = ax.imshow(w, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (token atendido)")
    ax.set_ylabel("Query (token que atende)")
    ax.set_title(title, fontsize=10)

    if annotate and n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if w[i, j] < 0.5 else "black")

    if created:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig
    return im


def plot_comparison(matrices, tokens, suptitle=None, annotate=True):
    """Matrices: lista de (titulo, tensor (T, T))."""
    n = len(matrices)
    t = len(tokens)
    fig, axes = plt.subplots(1, n, figsize=((0.7 * t + 3) * n, 0.7 * t + 3), squeeze=False)
    im = None
    for ax, (title, w) in zip(axes[0], matrices):
        im = plot_attention_matrix(w, tokens, title, ax=ax, annotate=annotate)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def plot_heads(weights, tokens, suptitle="Multi-Head Attention", annotate=False, max_cols=4):
    """weights: tensor (num_heads, T, T), uma matriz por head (ex.: weights[0] de um batch)."""
    h = weights.shape[0]
    cols = min(max_cols, h)
    rows = -(-h // cols)  # divisão com teto
    t = len(tokens)
    fig, axes = plt.subplots(rows, cols, figsize=((0.55 * t + 2.5) * cols, (0.55 * t + 2.5) * rows),
                             squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < h:
            plot_attention_matrix(weights[i], tokens, f"Head {i}", ax=ax, annotate=annotate)
        else:
            ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def save_figure(fig, path, dpi=150):
    """Salva a figura em `path` (cria as pastas se preciso) e a fecha."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import torch
    from causal_attention import CausalAttention

    torch.manual_seed(123)
    tokens = ["Your", "journey", "starts", "with", "one", "step"]
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], [0.55, 0.87, 0.66], [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33], [0.77, 0.25, 0.10], [0.05, 0.80, 0.55]]
    ).unsqueeze(0)

    ca = CausalAttention(3, 2, context_length=6)
    _, w = ca(inputs, return_weights=True)
    fig = plot_attention_matrix(w[0], tokens, "Causal Attention — 'Your journey starts with one step'")
    print("Figura salva em:", save_figure(fig, "demo_causal_attention.png"))
