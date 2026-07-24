"""
Genera las gráficas de resultados del simulador de reconexión.
Funciona directamente sobre los datos crudos (datos-salida_N.txt, hist_test_C.txt).

ESTRUCTURA ESPERADA DE CARPETAS:
    BASE_DIR/
    └── R{regla}/
        ├── RW/D2/{1..10}/datos-salida_N.txt
        ├── RWD/D2/{1..10}/datos-salida_N.txt
        ├── RWI/D2/{1..10}/datos-salida_N.txt
        └── N2Vp{p}q{q}/D2/{1..10}/datos-salida_N.txt

USO:
    Ajustar BASE_DIR, REGLAS, CICLOS y NODOS al inicio del script.
    python creaGraficas.py

SALIDA:
    BASE_DIR/Graficas/
        1_diametro/    2_dist_grados/    3_mapa_calor/    4_clustering_mod/
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = "." # Ruta base donde estan localizadas las carpetas de resultados
REGLAS      = [1,2,3]
CICLOS      = 30
NODOS       = 2500
EJECUCIONES = 10
LONG_ENLACE = "D2"
DPI         = 150

PQ_VALS = [0.25, 0.5, 1.0, 2.0]

# ─────────────────────────────────────────────────────────────────────────────
#  MAPEO DE NOMBRES DE CARPETA
# ─────────────────────────────────────────────────────────────────────────────
def fmt(v):
    s = str(int(v)) if v == int(v) else str(v)
    return s.replace(".", "_")

def carpeta_n2v(p, q):
    return f"N2Vp{fmt(p)}q{fmt(q)}"

def label_n2v(p, q):
    def fmtl(v): return str(int(v)) if v == int(v) else str(v)
    return f"p={fmtl(p)}, q={fmtl(q)}"

# ─────────────────────────────────────────────────────────────────────────────
#  PALETAS Y ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
COLORES_BASE = {"RW": "#2563eb", "RWD": "#16a34a", "RWI": "#dc2626"}

COLORES_Q = {
    0.25: "#e63946",   
    0.5:  "#2a9d8f",   
    1.0:  "#f4a261",   
    2.0:  "#7b2d8b",   
}

MARCADORES_Q = {
    0.25: "o",    
    0.5:  "s",    
    1.0:  "^",    
    2.0:  "D",    
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ─────────────────────────────────────────────────────────────────────────────
#  LECTURA DE DATOS CRUDOS
# ─────────────────────────────────────────────────────────────────────────────
def ruta_experimento(regla, algoritmo):
    return os.path.join(BASE_DIR, f"R{regla}", algoritmo, LONG_ENLACE)

def leer_datos_salida(path_txt):
    if not os.path.exists(path_txt): return None
    data = {}
    with open(path_txt) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split("\t")
            if len(parts) < 6: continue
            ciclo = int(parts[0])
            data[ciclo] = {
                "avCl": float(parts[1]),
                "dia":  float(parts[3]),
                "aspl": float(parts[4]),
            }
    return data

def promediar_ejecuciones(ruta_exp, metrica):
    todas = []
    for ejec in range(1, EJECUCIONES + 1):
        path = os.path.join(ruta_exp, str(ejec), f"datos-salida_{ejec}.txt")
        data = leer_datos_salida(path)
        if data is None: continue
        serie = []
        for c in range(CICLOS + 1):
            if c in data: serie.append(data[c][metrica])
            elif serie: serie.append(serie[-1])
            else: serie.append(0.0)
        todas.append(serie)
    if not todas: return None, None, None
    arr = np.array(todas)
    ciclos = list(range(CICLOS + 1))
    return ciclos, arr.mean(axis=0), arr.std(axis=0)

def leer_hist_final(ruta_exp, ejecucion=1):
    path = os.path.join(ruta_exp, str(ejecucion), f"hist_test_{CICLOS}.txt")
    if not os.path.exists(path): return None, None
    grados, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            g = int(parts[0])
            n = float(parts[1])
            if g > 0 and n > 0:
                grados.append(g)
                probs.append(n / NODOS)
    return np.array(grados), np.array(probs)

def leer_adjlist_final(ruta_exp, ejecucion=10):
    path = os.path.join(ruta_exp, str(ejecucion), f"graph_test_{CICLOS}.adjlist")
    if not os.path.exists(path): return None
    return nx.read_adjlist(path, nodetype=int)

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIONES DE PRE-EXTRACCIÓN
# ─────────────────────────────────────────────────────────────────────────────
def extraer_matrices_base():
    filas = ["RW", "RWD", "RWI"]
    mat_cl = np.full((len(filas), len(REGLAS)), np.nan)
    mat_mod = np.full((len(filas), len(REGLAS)), np.nan)

    for i, carpeta in enumerate(filas):
        for j, regla in enumerate(REGLAS):
            ruta = ruta_experimento(regla, carpeta)
            
            vals_cl = []
            for ejec in range(1, EJECUCIONES + 1):
                path = os.path.join(ruta, str(ejec), f"datos-salida_{ejec}.txt")
                data = leer_datos_salida(path)
                if data and CICLOS in data:
                    vals_cl.append(data[CICLOS]["avCl"])
            if vals_cl: mat_cl[i, j] = np.mean(vals_cl)

            G = leer_adjlist_final(ruta, ejecucion=1)
            if G is not None:
                try:
                    from networkx.algorithms.community import greedy_modularity_communities
                    from networkx.algorithms.community.quality import modularity
                    comms = greedy_modularity_communities(G)
                    mat_mod[i, j] = modularity(G, comms)
                except Exception:
                    pass
    return mat_cl, mat_mod

def extraer_matrices_pq(regla):
    n = len(PQ_VALS)
    mat_cl = np.full((n, n), np.nan)
    mat_mod = np.full((n, n), np.nan)

    for i, p in enumerate(PQ_VALS):
        for j, q in enumerate(PQ_VALS):
            carpeta = carpeta_n2v(p, q)
            ruta = ruta_experimento(regla, carpeta)
            
            vals_cl = []
            for ejec in range(1, EJECUCIONES + 1):
                path = os.path.join(ruta, str(ejec), f"datos-salida_{ejec}.txt")
                data = leer_datos_salida(path)
                if data and CICLOS in data:
                    vals_cl.append(data[CICLOS]["avCl"])
            if vals_cl: mat_cl[i, j] = np.mean(vals_cl)

            G = leer_adjlist_final(ruta, ejecucion=1)
            if G is not None:
                try:
                    from networkx.algorithms.community import greedy_modularity_communities
                    from networkx.algorithms.community.quality import modularity
                    comms = greedy_modularity_communities(G)
                    mat_mod[i, j] = modularity(G, comms)
                except Exception:
                    pass
    return mat_cl, mat_mod

# ─────────────────────────────────────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────
def grafica_diametro(regla, series, titulo, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in series:
        ruta = ruta_experimento(regla, s["carpeta"])
        ciclos, media, std = promediar_ejecuciones(ruta, "dia")
        if ciclos is None or media is None or std is None: continue
        
        marker = s.get("marker", "o")
        ax.plot(ciclos, media, color=s["color"], label=s["label"],
                linewidth=1.8, marker=marker, markersize=3.5, markevery=3)
        ax.fill_between(ciclos, media - std, media + std, color=s["color"], alpha=0.15)
        
    ax.set_xlabel("Ciclos de reconexión")
    ax.set_ylabel("Diámetro")
    ax.set_title(titulo, fontweight="bold", pad=12)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, framealpha=0.7)
        
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"{os.path.basename(out_path)}")

def grafica_dist_grados(regla, series, titulo, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in series:
        ruta = ruta_experimento(regla, s["carpeta"])
        todas_probs = {}
        for ejec in range(1, EJECUCIONES + 1):
            g, p = leer_hist_final(ruta, ejec)
            if g is None or p is None: continue
            for gi, pi in zip(g, p):
                todas_probs.setdefault(gi, []).append(pi)
        if not todas_probs: continue
        grados = sorted(todas_probs.keys())
        probs  = [np.mean(todas_probs[g]) for g in grados]
        stds   = [np.std(todas_probs[g])  for g in grados]
        
        marker = s.get("marker", "o")
        ax.errorbar(grados, probs, yerr=stds, color=s["color"], label=s["label"], 
                    marker=marker, markersize=3.5, linewidth=1.5, capsize=2, elinewidth=0.8)
                    
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k (grado)"); ax.set_ylabel("P(k)")
    ax.set_title(titulo, fontweight="bold", pad=12)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, framealpha=0.7)
        
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"{os.path.basename(out_path)}")

def grafica_mapa_calor(regla, s, titulo, out_path, vmin=None, vmax=None):
    ruta = ruta_experimento(regla, s["carpeta"])
    G = leer_adjlist_final(ruta, ejecucion=10)
    if G is None: return
    rows = cols = int(NODOS ** 0.5)
    mat = np.zeros((rows, cols))
    for nodo in G.nodes():
        idx = nodo - 1
        i, j = divmod(idx, cols)
        if 0 <= i < rows and 0 <= j < cols:
            mat[i, j] = G.degree(nodo)
            
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="plasma", aspect="equal", origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Grado del nodo")
    ax.set_title(titulo, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"{os.path.basename(out_path)}")

def renderizar_heatmap(mat, medida, fname, vmin, vmax, xticks_labels, yticks_labels, xlabel, ylabel, titulo):
    if np.all(np.isnan(mat)): return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(xticks_labels))); ax.set_xticklabels(xticks_labels)
    ax.set_yticks(range(len(yticks_labels))); ax.set_yticklabels(yticks_labels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold", pad=10)
    
    mid = (vmin + vmax) / 2
    for ii in range(len(yticks_labels)):
        for jj in range(len(xticks_labels)):
            if not np.isnan(mat[ii, jj]):
                color_txt = "white" if mat[ii, jj] > mid else "black"
                ax.text(jj, ii, f"{mat[ii,jj]:.3f}", ha="center", va="center",
                        fontsize=8 if len(xticks_labels) < 4 else 7, color=color_txt)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)
    print(f"{os.path.basename(fname)}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    GRAFICAS_DIR = os.path.join(BASE_DIR, "graficas")
    dir1 = os.path.join(GRAFICAS_DIR, "1_diametro");    os.makedirs(dir1, exist_ok=True)
    dir2 = os.path.join(GRAFICAS_DIR, "2_dist_grados"); os.makedirs(dir2, exist_ok=True)
    dir3 = os.path.join(GRAFICAS_DIR, "3_mapa_calor");  os.makedirs(dir3, exist_ok=True)
    dir4 = os.path.join(GRAFICAS_DIR, "4_clustering_mod"); os.makedirs(dir4, exist_ok=True)

    series_base = [
        {"label": "RW",  "carpeta": "RW",  "color": COLORES_BASE["RW"], "marker": "o"},
        {"label": "RWD", "carpeta": "RWD", "color": COLORES_BASE["RWD"], "marker": "s"},
        {"label": "RWI", "carpeta": "RWI", "color": COLORES_BASE["RWI"], "marker": "^"},
    ]

    mat_cl_base, mat_mod_base = extraer_matrices_base()
    
    matrices_pq_cl, matrices_pq_mod = {}, {}
    for regla in REGLAS:
        cl, mod = extraer_matrices_pq(regla)
        matrices_pq_cl[regla] = cl
        matrices_pq_mod[regla] = mod

    # Límite global Clustering
    todas_cl = [mat_cl_base] + list(matrices_pq_cl.values())
    validas_cl = [m for m in todas_cl if not np.isnan(m).all()]
    if validas_cl:
        vmin_cl, vmax_cl = np.nanmin([np.nanmin(m) for m in validas_cl]), np.nanmax([np.nanmax(m) for m in validas_cl])
    else: vmin_cl, vmax_cl = 0, 1

    # Límite global Modularidad
    todas_mod = [mat_mod_base] + list(matrices_pq_mod.values())
    validas_mod = [m for m in todas_mod if not np.isnan(m).all()]
    if validas_mod:
        vmin_mod, vmax_mod = np.nanmin([np.nanmin(m) for m in validas_mod]), np.nanmax([np.nanmax(m) for m in validas_mod])
    else: vmin_mod, vmax_mod = 0, 1

    # Límite global Grados
    g_min, g_max = float('inf'), float('-inf')
    for regla in REGLAS:
        # Base
        for s in series_base:
            G = leer_adjlist_final(ruta_experimento(regla, s["carpeta"]), ejecucion=10)
            if G:
                grados = [d for n, d in G.degree()]
                if grados: g_min, g_max = min(g_min, min(grados)), max(g_max, max(grados))
        # Node2Vec
        for p in PQ_VALS:
            for q in PQ_VALS:
                G = leer_adjlist_final(ruta_experimento(regla, carpeta_n2v(p, q)), ejecucion=10)
                if G:
                    grados = [d for n, d in G.degree()]
                    if grados: g_min, g_max = min(g_min, min(grados)), max(g_max, max(grados))
                    
    if g_min == float('inf'): g_min, g_max = 0, 1

    # --- CREACIÓN DE GRÁFICAS ---
    print("\n[4] Heatmap base — RW/RWD/RWI × Reglas (Clustering y Modularidad)")
    col_labels = [f"R{r}" for r in REGLAS]
    row_labels = ["RW", "RWD", "RWI"]
    
    renderizar_heatmap(mat_cl_base, "Coeficiente de Agrupamiento", 
                       os.path.join(dir4, "heatmap_clustering_base.png"), vmin_cl, vmax_cl, 
                       col_labels, row_labels, "Regla", "Algoritmo", f"Coef. de Agrupamiento (ciclo {CICLOS}) — RW / RWD / RWI")
                       
    renderizar_heatmap(mat_mod_base, "Modularidad", 
                       os.path.join(dir4, "heatmap_modularidad_base.png"), vmin_mod, vmax_mod, 
                       col_labels, row_labels, "Regla", "Algoritmo", f"Modularidad (ciclo {CICLOS}) — RW / RWD / RWI")

    for regla in REGLAS:
        print(f"\n── Regla {regla} ──────────────────────────────")

        print("  [1] Diámetro — RW, RWD, RWI")
        grafica_diametro(regla, series_base, f"Evolución del diámetro — R{regla} (RW, RWD, RWI)",
                         os.path.join(dir1, f"diametro_R{regla}_base.png"))

        print("  [2] Dist. grados — RW, RWD, RWI")
        grafica_dist_grados(regla, series_base, f"Distribución de grados — R{regla} (RW, RWD, RWI)",
                            os.path.join(dir2, f"dist_grados_R{regla}_base.png"))

        for s in series_base:
            print(f"  [3] Mapa calor — {s['label']}")
            grafica_mapa_calor(regla, s, f"Mapa de grados — R{regla} {s['label']} (ciclo {CICLOS}, ejec. 10)",
                               os.path.join(dir3, f"mapa_calor_R{regla}_{s['carpeta']}.png"), vmin=g_min, vmax=g_max)

        for p in PQ_VALS:
            series_n2v = [
                {
                    "label": label_n2v(p, q), 
                    "carpeta": carpeta_n2v(p, q), 
                    "color": COLORES_Q[q],
                    "marker": MARCADORES_Q[q]
                }
                for q in PQ_VALS
            ]
            sfx = f"R{regla}_p{fmt(p)}"

            print(f"  [1] Diámetro — N2V p={p}")
            grafica_diametro(regla, series_n2v, f"Evolución del diámetro — R{regla} Node2Vec p={p}",
                             os.path.join(dir1, f"diametro_{sfx}_n2v.png"))

            print(f"  [2] Dist. grados — N2V p={p}")
            grafica_dist_grados(regla, series_n2v, f"Distribución de grados — R{regla} Node2Vec p={p}",
                                os.path.join(dir2, f"dist_grados_{sfx}_n2v.png"))

            print(f"  [3] Mapa calor — N2V p={p}, q={PQ_VALS[0]}")
            grafica_mapa_calor(regla, series_n2v[0], f"Mapa de grados — R{regla} N2V p={p}, q={PQ_VALS[0]} (ciclo {CICLOS}, ejec. 10)",
                               os.path.join(dir3, f"mapa_calor_{sfx}_n2v.png"), vmin=g_min, vmax=g_max)

        print("  [4] Heatmap p×q (Node2Vec) — Clustering y Modularidad")
        labels_pq = [str(int(v)) if v == int(v) else str(v) for v in PQ_VALS]
        renderizar_heatmap(matrices_pq_cl[regla], "Coeficiente de Agrupamiento",
                           os.path.join(dir4, f"heatmap_clustering_R{regla}.png"), vmin_cl, vmax_cl, 
                           labels_pq, labels_pq, "q", "p", f"R{regla} — Coef. Agrupamiento (ciclo {CICLOS})")
                           
        renderizar_heatmap(matrices_pq_mod[regla], "Modularidad",
                           os.path.join(dir4, f"heatmap_modularidad_R{regla}.png"), vmin_mod, vmax_mod, 
                           labels_pq, labels_pq, "q", "p", f"R{regla} — Modularidad (ciclo {CICLOS})")

    print(f"\n Gráficas guardadas en: {GRAFICAS_DIR}")

if __name__ == "__main__":
    main()