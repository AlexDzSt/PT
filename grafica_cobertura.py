"""
grafica_cobertura.py
─────────────────────
Genera la gráfica:
  eje x → longitud relativa  (pasos / N)
  eje y → cobertura          (nodos únicos / N)

Para cada estrategia se ejecutan N_WALKS caminatas independientes y se
muestra la media ± desviación estándar, interpoladas sobre una rejilla
común en x.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from caminantes import (
    jump_prob_traditional, jump_prob_degree_biased,
    jump_prob_inverse_degree, jump_prob_node2vec,
    walk,
)
from datos_caminata import curva_cobertura

STRATEGIES: dict = {
    "Traditional RW":    jump_prob_traditional,
    "Degree-Biased RW":  jump_prob_degree_biased,
    "Inverse-Degree RW": jump_prob_inverse_degree,
    "Node2Vec RW":       jump_prob_node2vec,
}

STRATEGY_COLORS: dict = {
    "Traditional RW":    "#4FC3F7",
    "Degree-Biased RW":  "#81C784",
    "Inverse-Degree RW": "#FFB74D",
    "Node2Vec RW":       "#CE93D8",
}

# ─────────────────────────────────────────────────────────────────────────────
#  PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

G          = nx.karate_club_graph()   # cambia aquí el grafo si lo necesitas
N          = G.number_of_nodes()
START      = 0                        # nodo inicial
MAX_STEPS  = N * 10                   # suficiente para cubrir el grafo
STOP_PROB  = 0.0                      # 0 → siempre corre MAX_STEPS pasos
P, Q       = 2.0, 0.5                 # parámetros de node2vec
N_WALKS    = 200                      # número de caminatas por estrategia
N_GRID     = 300                      # puntos en la rejilla común de x

# ─────────────────────────────────────────────────────────────────────────────
#  CÓMPUTO
# ─────────────────────────────────────────────────────────────────────────────

# La rejilla en x va de 1/N (primer paso) hasta MAX_STEPS/N
x_grid = np.linspace(1 / N, MAX_STEPS / N, N_GRID)

resultados: dict = {}   # strategy → (y_mean, y_std)

for strategy, jump_fn in STRATEGIES.items():
    y_curves = []
    for _ in range(N_WALKS):
        path  = walk(G, START, jump_fn, STOP_PROB, P, Q, MAX_STEPS)
        curve = curva_cobertura(path, N)

        xs = np.array([pt[0] for pt in curve])
        ys = np.array([pt[1] for pt in curve])

        # Interpolar sobre la rejilla común; fuera del rango se satura
        y_interp = np.interp(x_grid, xs, ys,
                             left=ys[0], right=ys[-1])
        y_curves.append(y_interp)

    mat             = np.array(y_curves)          # (N_WALKS, N_GRID)
    resultados[strategy] = (mat.mean(axis=0), mat.std(axis=0))

# ─────────────────────────────────────────────────────────────────────────────
#  GRÁFICA
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

for strategy, (y_mean, y_std) in resultados.items():
    color = STRATEGY_COLORS[strategy]
    ax.plot(x_grid, y_mean,
            color=color, linewidth=2.2, label=strategy)
    ax.fill_between(x_grid,
                    y_mean - y_std,
                    y_mean + y_std,
                    color=color, alpha=0.15)

# Línea de referencia: cobertura máxima = 1.0
ax.axhline(1.0, color="#546E7A", linewidth=0.8, linestyle="--")

ax.set_xlabel("Longitud relativa  (pasos / N)", color="#90A4AE", fontsize=11)
ax.set_ylabel("Cobertura  (nodos únicos / N)",  color="#90A4AE", fontsize=11)
ax.set_title(
    f"Cobertura vs Longitud relativa\n"
    f"Grafo: Karate Club  ·  N={N}  ·  inicio={START}  "
    f"·  p={P}  q={Q}  ·  {N_WALKS} caminatas/estrategia",
    color="white", fontsize=11, pad=10
)

ax.tick_params(colors="#90A4AE", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#263238")

ax.set_xlim(left=0)
ax.set_ylim(0, 1.08)

legend = ax.legend(
    facecolor="#1E2A38", edgecolor="#37474F",
    labelcolor="white", fontsize=9, loc="lower right"
)

fig.tight_layout()
plt.savefig("cobertura_vs_longitud.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Gráfica guardada en cobertura_vs_longitud.png")