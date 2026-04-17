import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from app.utils.aux_functions import (
    preprocess,
    compute_iptw_weights,
    compute_smd,
    ate_iptw,
    bootstrap_ci
)


def plot_dag(arestas, confundidores, tratamentos, outcomes,
             titulo="DAG Causal", figsize=(16, 10), destaque_t=None, destaque_o=None):
    """
    Plota o DAG com layout em camadas:
      Camada 1 (topo): confundidores
      Camada 2 (meio): tratamentos
      Camada 3 (base): outcomes
    """
    G = nx.DiGraph()
    G.add_edges_from(arestas)

    # Layout em camadas
    pos = {}
    n_conf = len(confundidores)
    for i, n in enumerate(confundidores):
        pos[n] = (i * 2.0, 2)
    for i, n in enumerate(tratamentos):
        pos[n] = (i * 4.0 + 2, 1)
    for i, n in enumerate(outcomes):
        pos[n] = (i * 6.0 + 4, 0)

    # Cores dos nós
    cor_map = {}
    for n in G.nodes():
        if n in outcomes:
            cor_map[n] = "#1565c0"
        elif n in tratamentos:
            cor_map[n] = "#2e7d32" if destaque_t is None or n == destaque_t else "#a5d6a7"
        else:
            cor_map[n] = "#e65100"

    # Destaque do outcome
    if destaque_o:
        for n in outcomes:
            if n != destaque_o:
                cor_map[n] = "#90caf9"

    cores = [cor_map.get(n, "gray") for n in G.nodes()]

    # Cores das arestas
    edge_colors = []
    for u, v in G.edges():
        if u in tratamentos and v in outcomes:
            edge_colors.append("#1b5e20")
        elif u in confundidores:
            edge_colors.append("#bf360c")
        else:
            edge_colors.append("#455a64")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=cores, node_size=2200,
        font_size=8, font_color="white", font_weight="bold",
        edge_color=edge_colors, arrows=True,
        arrowsize=18, arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        width=1.5
    )

    # Legenda
    legend = [
        mpatches.Patch(color="#e65100", label="Confundidor"),
        mpatches.Patch(color="#2e7d32", label="Tratamento"),
        mpatches.Patch(color="#1565c0", label="Outcome"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title(titulo, fontsize=13, pad=15)
    ax.axis("off")
    plt.tight_layout()
    return fig




def run_iptw_analysis(df, treatment, outcome, confundidores,
                      trim_percentile=99, assoc_bruta=None, n_bootstrap=200):
    print(f"{'='*60}")
    print(f"IPTW v10: {treatment} -> {outcome}")
    print(f"{'='*60}")

    df_model = preprocess(df, treatment, outcome, confundidores)
    print(f"N apos dropna: {len(df_model):,}")

    ps, weights, auc = compute_iptw_weights(
        df_model, treatment, confundidores, trim_percentile=trim_percentile
    )
    print(f"\nAUC-ROC PS: {auc:.4f}")
    print(f"PS     — media: {ps.mean():.3f} | min: {ps.min():.3f} | max: {ps.max():.3f}")
    print(f"Pesos  — media: {weights.mean():.3f} | min: {weights.min():.3f} | max: {weights.max():.3f}")

    smd_antes  = compute_smd(df_model, treatment, confundidores)
    smd_depois = compute_smd(df_model, treatment, confundidores, weights=weights)
    df_smd = pd.DataFrame({
        'SMD antes':  smd_antes.round(4),
        'SMD depois': smd_depois.round(4),
        'Balanceado?': smd_depois.apply(lambda x: 'OK' if x < 0.1 else 'FAIL')
    })
    print("\n--- Balanco das covariaveis (SMD < 0.1 = OK) ---")
    print(df_smd.to_string())

    ate = ate_iptw(df_model, treatment, outcome, weights)
    print(f"\nAssoc. bruta EDA: {assoc_bruta:+.4f}" if assoc_bruta is not None else "")
    print(f"ATE (IPTW v10):   {ate:+.6f}")

    print(f"Calculando IC via bootstrap ({n_bootstrap} amostras)...")
    lower, upper, boot_ates = bootstrap_ci(
        df_model, treatment, outcome, confundidores,
        n_bootstrap=n_bootstrap, trim_percentile=trim_percentile
    )
    print(f"IC 95%: [{lower:.6f}, {upper:.6f}]")
    significativo = (lower > 0 or upper < 0)
    print(f"Resultado: {'Significativo' if significativo else 'NAO significativo (IC contem zero)'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(boot_ates, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(ate,   color='red',    linewidth=2,   label=f'ATE={ate:.4f}')
    axes[0].axvline(lower, color='orange', linewidth=1.5, linestyle='--', label=f'IC inf={lower:.4f}')
    axes[0].axvline(upper, color='orange', linewidth=1.5, linestyle='--', label=f'IC sup={upper:.4f}')
    axes[0].axvline(0,     color='black',  linewidth=1,   linestyle=':')
    axes[0].set_title(f'Bootstrap ATE\n{treatment} -> {outcome}')
    axes[0].set_xlabel('ATE estimado')
    axes[0].set_ylabel('Frequencia')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    xi = np.arange(len(confundidores))
    axes[1].barh(xi - 0.2, smd_antes.values,  0.4, label='Antes',  color='coral',     alpha=0.8)
    axes[1].barh(xi + 0.2, smd_depois.values, 0.4, label='Depois', color='steelblue', alpha=0.8)
    axes[1].axvline(0.1, color='red', linestyle='--', linewidth=1, label='Limite 0.1')
    axes[1].set_yticks(xi)
    axes[1].set_yticklabels(confundidores, fontsize=8)
    axes[1].set_xlabel('SMD')
    axes[1].set_title('Balanco das covariaveis')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'../../reports/figures/iptw_v10_{treatment}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return {
        'treatment': treatment, 'outcome': outcome,
        'N': len(df_model), 'auc': auc,
        'assoc_bruta': assoc_bruta,
        'ate': ate, 'ic_lower': lower, 'ic_upper': upper,
        'significativo': significativo,
    }