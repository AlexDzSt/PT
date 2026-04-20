import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import random
from collections import Counter

from caminantes import (
    jump_prob_traditional, jump_prob_degree_biased,
    jump_prob_inverse_degree, jump_prob_node2vec,
    walk, step_probabilities,
)

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
#  CONSTANTES DE ESTILO (Entry widgets)
# ─────────────────────────────────────────────────────────────────────────────

ENTRY_KW = dict(
    bg="#1E2A38", fg="#E8F5E9",
    insertbackground="#E8F5E9",
    relief="flat",
    highlightthickness=1,
    highlightbackground="#37474F",
    highlightcolor="#80CBC4",
    font=("Consolas", 10),
)
DIS_BG = "#161B22"
DIS_FG = "#455A64"


# ─────────────────────────────────────────────────────────────────────────────
#  DIBUJO DE RED
# ─────────────────────────────────────────────────────────────────────────────

def draw_network(
    ax,
    G:            nx.Graph,
    pos:          dict,
    path:         list  = None,
    start_node          = None,
    strategy:     str   = "Traditional RW",
    show_probs:   bool  = False,
    current_probs: dict = None,
) -> None:
    """
    Dibuja G sobre ax con la caminata path resaltada.

    - Nodo de inicio   → amarillo   (#F9A825)
    - Nodo actual      → rojo       (#EF5350)
    - Nodos visitados  → color de la estrategia, intensidad ∝ frecuencia
    - Aristas recorridas → color de la estrategia
    - Si show_probs y current_probs: etiquetas de probabilidad sobre aristas
      del nodo actual hacia sus vecinos.

    current_probs debe corresponder a las probabilidades DESDE el nodo
    actual (path[-1]) para que las etiquetas sean coherentes.
    """
    ax.clear()
    ax.set_facecolor("#0D1117")

    # ── Colores y tamaños de nodos ──
    nc  = {n: "#263238" for n in G.nodes()}
    ns  = {n: 300       for n in G.nodes()}
    vc  = Counter(path) if path else {}

    if vc:
        mx  = max(vc.values())
        sc  = STRATEGY_COLORS.get(strategy, "#4FC3F7")
        r_, g_, b_ = int(sc[1:3], 16), int(sc[3:5], 16), int(sc[5:7], 16)
        for node, cnt in vc.items():
            inten   = 0.3 + 0.7 * (cnt / mx)
            nc[node] = "#{:02x}{:02x}{:02x}".format(
                int(r_ * inten), int(g_ * inten), int(b_ * inten))
            ns[node] = 300 + 400 * (cnt / mx)

    # Nodo de inicio siempre amarillo (sobreescribe el color de visitado)
    if start_node is not None and start_node in G:
        nc[start_node] = "#F9A825"
        ns[start_node] = 700

    # Nodo actual (último de la caminata) siempre rojo
    if path:
        nc[path[-1]] = "#EF5350"
        ns[path[-1]] = 700

    node_colors = [nc[n] for n in G.nodes()]
    node_sizes  = [ns[n] for n in G.nodes()]

    # ── Aristas ──
    walk_edges = set()
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            walk_edges.add((min(a, b), max(a, b)))

    strat_color = STRATEGY_COLORS.get(strategy, "#4FC3F7")
    edge_colors, edge_widths = [], []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key in walk_edges:
            edge_colors.append(strat_color)
            edge_widths.append(2.5)
        else:
            edge_colors.append("#37474F")
            edge_widths.append(0.8)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, linewidths=1.5,
                           edgecolors="#546E7A")
    nx.draw_networkx_labels(G, pos, {n: str(n) for n in G.nodes()},
                            ax=ax, font_color="white",
                            font_size=7, font_weight="bold")

    # ── Etiquetas de probabilidad (desde el nodo actual) ──
    if show_probs and current_probs and path:
        cur = path[-1]
        edge_label_dict = {(cur, nb): f"{pv:.2f}"
                           for nb, pv in current_probs.items()}
        nx.draw_networkx_edge_labels(
            G, pos, edge_label_dict, ax=ax,
            font_color="#FFF176", font_size=7,
            bbox=dict(boxstyle="round,pad=0.1", fc="#1A237E", alpha=0.7))

    ax.set_title(f"Red — {strategy}", color="white",
                 fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
#  APLICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    """Ventana principal del explorador de caminatas aleatorias."""

    GRAPHS = {
        "Zachary's Karate Club":  "karate",
        "Petersen Graph":         "petersen",
        "Barbell (5-5)":          "barbell",
        "Erdős–Rényi (30,0.15)":  "er",
        "Barabási–Albert (30,2)": "ba",
        "Grid 5×5":               "grid",
    }

    def __init__(self):
        super().__init__()
        self.title("Random Walk Node Discovery Explorer")
        self.configure(bg="#0D1117")
        self.minsize(1150, 720)

        self.G:             nx.Graph = None
        self.pos:           dict     = None
        # current_path  : lista de nodos de la caminata activa
        # current_probs : P(salto) desde path[-1] hacia sus vecinos
        #                 Siempre corresponde al NODO ACTUAL (path[-1]).
        self.current_path:  list = []
        self.current_probs: dict = {}

        self._build_ui()
        self._load_graph("karate")

    # ═══════════════════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE LA UI
    # ═══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        base_font  = ("Consolas", 10)
        title_font = ("Consolas", 11, "bold")
        hdr_font   = ("Consolas", 9, "bold")

        style = ttk.Style(self)
        style.theme_use("clam")

        # Frames y labels
        style.configure("TFrame",        background="#0D1117")
        style.configure("TLabel",        background="#0D1117",
                        foreground="#ECEFF1", font=base_font)
        style.configure("Header.TLabel", background="#0D1117",
                        foreground="#80CBC4", font=hdr_font)
        style.configure("Dim.TLabel",    background="#0D1117",
                        foreground="#546E7A", font=("Consolas", 8))

        # Combobox legible (fondo oscuro, texto claro en lista desplegable)
        style.configure("TCombobox",
                        background="#1E2A38", foreground="#E8F5E9",
                        fieldbackground="#1E2A38",
                        selectbackground="#37474F", selectforeground="#E8F5E9",
                        arrowcolor="#80CBC4", insertcolor="#E8F5E9",
                        font=base_font)
        style.map("TCombobox",
                  fieldbackground=[("readonly","#1E2A38"),("disabled","#161B22"),
                                   ("active","#1E2A38")],
                  foreground=[("readonly","#E8F5E9"),("disabled","#546E7A"),
                               ("active","#E8F5E9")],
                  selectforeground=[("readonly","#E8F5E9")],
                  selectbackground=[("readonly","#263238")])
        self.option_add("*TCombobox*Listbox.background",       "#1E2A38")
        self.option_add("*TCombobox*Listbox.foreground",       "#E8F5E9")
        self.option_add("*TCombobox*Listbox.selectBackground", "#37474F")
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
        self.option_add("*TCombobox*Listbox.font",             "Consolas 10")

        # Botones
        for name, bg, abg, fnt, pad in [
            ("Run",     "#00695C", "#00897B", title_font, 8),
            ("Step",    "#1A237E", "#283593", base_font,  6),
            ("Clear",   "#4E342E", "#6D4C41", base_font,  6),
            ("Compare", "#4A148C", "#6A1B9A", base_font,  6),
        ]:
            style.configure(f"{name}.TButton", background=bg, foreground="white",
                            font=fnt, padding=pad, relief="flat")
            style.map(f"{name}.TButton", background=[("active", abg)])

        style.configure("TCheckbutton", background="#0D1117",
                        foreground="#ECEFF1", font=base_font)
        style.configure("TSeparator", background="#263238")

        # Layout principal: panel izquierdo + notebook derecho
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=8)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, width=285)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.pack_propagate(False)
        self._build_controls(left)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        tab_graph   = ttk.Frame(self.notebook)
        tab_probs   = ttk.Frame(self.notebook)
        tab_compare = ttk.Frame(self.notebook)
        tab_log     = ttk.Frame(self.notebook)

        self.notebook.add(tab_graph,   text="  Red  ")
        self.notebook.add(tab_probs,   text="  Probabilidades  ")
        self.notebook.add(tab_compare, text="  Comparar Estrategias  ")
        self.notebook.add(tab_log,     text="  Log  ")

        self._build_graph_tab(tab_graph)
        self._build_probs_tab(tab_probs)
        self._build_compare_tab(tab_compare)
        self._build_log_tab(tab_log)

    # ── Panel de controles ───────────────────────────────────────────────────

    def _build_controls(self, parent):
        pad  = {"padx": 8, "pady": 3}
        pad1 = {"padx": 8, "pady": 1}

        # Scrollable inner frame
        cv = tk.Canvas(parent, bg="#0D1117", highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(cv)
        wid   = cv.create_window((0, 0), window=inner, anchor="nw")
        cv.bind("<Configure>",    lambda e: cv.itemconfig(wid, width=e.width))
        inner.bind("<Configure>", lambda e: cv.configure(
            scrollregion=cv.bbox("all")))

        # Helper: fila label + Entry + hint opcional
        def param_row(label_text, hint, sv, width=9):
            frm = ttk.Frame(inner)
            frm.pack(fill="x", **pad1)
            ttk.Label(frm, text=label_text, width=17,
                      font=("Consolas", 9)).pack(side="left")
            e = tk.Entry(frm, textvariable=sv, width=width, **ENTRY_KW)
            e.pack(side="left", padx=(0, 4))
            if hint:
                ttk.Label(frm, text=hint, style="Dim.TLabel").pack(side="left")
            return e

        # ── Grafo ──
        ttk.Label(inner, text="GRAFO", style="Header.TLabel").pack(anchor="w", **pad)
        self.graph_var = tk.StringVar(value="Zachary's Karate Club")
        cb_g = ttk.Combobox(inner, textvariable=self.graph_var,
                             values=list(self.GRAPHS.keys()),
                             state="readonly", width=28)
        cb_g.pack(**pad, fill="x")
        cb_g.bind("<<ComboboxSelected>>",
                  lambda e: self._load_graph(self.GRAPHS[self.graph_var.get()]))

        ttk.Separator(inner).pack(fill="x", pady=5)

        # ── Estrategia ──
        ttk.Label(inner, text="ESTRATEGIA", style="Header.TLabel").pack(anchor="w", **pad)
        self.strategy_var = tk.StringVar(value="Traditional RW")
        cb_s = ttk.Combobox(inner, textvariable=self.strategy_var,
                             values=list(STRATEGIES.keys()),
                             state="readonly", width=28)
        cb_s.pack(**pad, fill="x")
        cb_s.bind("<<ComboboxSelected>>", lambda e: self._on_strategy_change())

        ttk.Separator(inner).pack(fill="x", pady=5)

        # ── Parámetros ──
        ttk.Label(inner, text="PARÁMETROS", style="Header.TLabel").pack(anchor="w", **pad)

        self.start_var = tk.StringVar(value="0")
        param_row("Nodo inicio:", "entero", self.start_var)

        self.stop_var = tk.StringVar(value="0.15")
        param_row("P(parada):", "(0, 1)", self.stop_var)

        self.p_var    = tk.StringVar(value="1.0")
        self.entry_p  = param_row("p  (node2vec):", "> 0", self.p_var)

        self.q_var    = tk.StringVar(value="1.0")
        self.entry_q  = param_row("q  (node2vec):", "> 0", self.q_var)

        self.maxsteps_var = tk.StringVar(value="50")
        param_row("Máx. pasos:", "entero", self.maxsteps_var)

        self.show_probs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(inner, text="Mostrar probs en red",
                        variable=self.show_probs_var,
                        command=self._refresh_graph).pack(anchor="w", **pad)

        ttk.Separator(inner).pack(fill="x", pady=6)

        # ── Botones ──
        ttk.Button(inner, text="▶  EJECUTAR CAMINATA", style="Run.TButton",
                   command=self._run_walk).pack(fill="x", **pad)
        ttk.Button(inner, text="⊕  PASO A PASO",       style="Step.TButton",
                   command=self._step_walk).pack(fill="x", **pad)
        ttk.Button(inner, text="⊞  COMPARAR TODAS",    style="Compare.TButton",
                   command=self._run_compare).pack(fill="x", **pad)
        ttk.Button(inner, text="✕  LIMPIAR",            style="Clear.TButton",
                   command=self._clear).pack(fill="x", **pad)

        ttk.Separator(inner).pack(fill="x", pady=6)

        # ── Estadísticas ──
        ttk.Label(inner, text="ESTADÍSTICAS", style="Header.TLabel").pack(anchor="w", **pad)
        self.stats_text = tk.Text(inner, height=9, width=30, state="disabled",
                                  bg="#161B22", fg="#A5D6A7",
                                  font=("Consolas", 9), relief="flat")
        self.stats_text.pack(**pad, fill="x")

        # ── Fórmula activa ──
        ttk.Separator(inner).pack(fill="x", pady=4)
        ttk.Label(inner, text="FÓRMULA ACTIVA", style="Header.TLabel").pack(anchor="w", **pad)
        self.formula_text = tk.Text(inner, height=7, width=30, state="disabled",
                                    bg="#161B22", fg="#CE93D8",
                                    font=("Consolas", 8), wrap="word", relief="flat")
        self.formula_text.pack(**pad, fill="x")
        self._update_formula()
        self._update_pq_state()

    # ── Tabs ─────────────────────────────────────────────────────────────────

    def _build_graph_tab(self, parent):
        self.fig = Figure(figsize=(7, 5), facecolor="#0D1117")
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0D1117")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tf = ttk.Frame(parent)
        tf.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tf)

    def _build_probs_tab(self, parent):
        self.prob_fig = Figure(figsize=(7, 5), facecolor="#0D1117")
        self.prob_ax  = self.prob_fig.add_subplot(111)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=parent)
        self.prob_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_probs_empty("Sin caminata activa")

    def _build_compare_tab(self, parent):
        self.cmp_fig    = Figure(figsize=(10, 7), facecolor="#0D1117")
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_fig, master=parent)
        self.cmp_canvas.get_tk_widget().pack(fill="both", expand=True)

        bot = ttk.Frame(parent)
        bot.pack(fill="x", padx=6, pady=4)
        self.cmp_metrics = tk.Text(bot, height=5, state="disabled",
                                   bg="#161B22", fg="#B0BEC5",
                                   font=("Consolas", 9), relief="flat")
        self.cmp_metrics.pack(fill="x")

        # Mensaje inicial
        self.cmp_fig.text(0.5, 0.5,
                          "Pulsa  ⊞ COMPARAR TODAS  para ejecutar\n"
                          "las 4 estrategias con los parámetros actuales",
                          ha="center", va="center",
                          color="#546E7A", fontsize=13, fontfamily="monospace")
        self.cmp_canvas.draw()

    def _build_log_tab(self, parent):
        self.log_text = scrolledtext.ScrolledText(
            parent, state="disabled", bg="#161B22", fg="#B0BEC5",
            font=("Consolas", 9), relief="flat")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text.tag_config("start",   foreground="#F9A825",
                                 font=("Consolas", 9, "bold"))
        self.log_text.tag_config("end",     foreground="#EF5350",
                                 font=("Consolas", 9, "bold"))
        self.log_text.tag_config("header",  foreground="#80CBC4",
                                 font=("Consolas", 9, "bold"))
        self.log_text.tag_config("normal",  foreground="#B0BEC5")
        self.log_text.tag_config("step",    foreground="#CE93D8")
        self.log_text.tag_config("compare", foreground="#FFB74D",
                                 font=("Consolas", 9, "bold"))

    # ═══════════════════════════════════════════════════════════════════════
    #  CARGA DE GRAFOS
    # ═══════════════════════════════════════════════════════════════════════

    def _load_graph(self, key: str):
        if key == "karate":
            G = nx.karate_club_graph()
        elif key == "petersen":
            G = nx.petersen_graph()
        elif key == "barbell":
            G = nx.barbell_graph(5, 5)
        elif key == "er":
            G = nx.erdos_renyi_graph(30, 0.15, seed=42)
        elif key == "ba":
            G = nx.barabasi_albert_graph(30, 2, seed=42)
        else:  # grid
            raw = nx.grid_2d_graph(5, 5)
            G   = nx.relabel_nodes(raw, {n: i for i, n in enumerate(raw.nodes())})

        self.G   = G
        self.pos = nx.spring_layout(G, seed=42)
        self._clear_state()

        # Ajustar nodo de inicio si ya no existe en el nuevo grafo
        try:
            sn = int(self.start_var.get())
            if sn not in G.nodes():
                self.start_var.set(str(min(G.nodes())))
        except (ValueError, AttributeError):
            self.start_var.set(str(min(G.nodes())))

        self._refresh_graph()
        self._update_stats()
        self._log_line(
            f"Grafo cargado: {self.graph_var.get()} "
            f"({G.number_of_nodes()} nodos, {G.number_of_edges()} aristas)\n",
            "header")

    def _clear_state(self):
        """Limpia la caminata activa sin redibujar (usado al cargar grafo)."""
        self.current_path  = []
        self.current_probs = {}

    # ═══════════════════════════════════════════════════════════════════════
    #  VALIDACIÓN DE PARÁMETROS
    # ═══════════════════════════════════════════════════════════════════════

    def _get_params(self) -> dict | None:
        """
        Lee y valida los campos de entrada.
        Retorna un dict con las claves: start, stop_prob, p, q, max_steps
        o None si hay errores de validación.
        """
        errors = []

        try:
            start = int(self.start_var.get())
            if start not in self.G.nodes():
                errors.append(f"Nodo inicio '{start}' no existe en el grafo.")
        except ValueError:
            errors.append("Nodo inicio debe ser un entero.")
            start = None

        try:
            stop_prob = float(self.stop_var.get())
            if not (0.0 < stop_prob < 1.0):
                errors.append("P(parada) debe estar en (0, 1).")
        except ValueError:
            errors.append("P(parada) debe ser un número decimal.")
            stop_prob = None

        try:
            p = float(self.p_var.get())
            if p <= 0:
                errors.append("p debe ser > 0.")
        except ValueError:
            errors.append("p debe ser un número decimal.")
            p = None

        try:
            q = float(self.q_var.get())
            if q <= 0:
                errors.append("q debe ser > 0.")
        except ValueError:
            errors.append("q debe ser un número decimal.")
            q = None

        try:
            max_steps = int(float(self.maxsteps_var.get()))
            if max_steps < 1:
                errors.append("Máx. pasos debe ser ≥ 1.")
        except ValueError:
            errors.append("Máx. pasos debe ser un entero.")
            max_steps = None

        if errors:
            messagebox.showerror("Error de parámetros", "\n".join(errors))
            return None

        return dict(start=start, stop_prob=stop_prob, p=p, q=q,
                    max_steps=max_steps)

    # ═══════════════════════════════════════════════════════════════════════
    #  ACCIONES PRINCIPALES
    # ═══════════════════════════════════════════════════════════════════════

    def _run_walk(self):
        """Ejecuta la caminata completa y actualiza todas las vistas."""
        params = self._get_params()
        if params is None:
            return
        strategy = self.strategy_var.get()

        # 1. Ejecutar la caminata
        self.current_path = walk(
            self.G, params["start"], STRATEGIES[strategy],
            params["stop_prob"], params["p"], params["q"],
            params["max_steps"])

        # 2. Calcular probs DESDE el nodo final (path[-1])
        #    step_probabilities usa path[-2] como source para node2vec
        self.current_probs = step_probabilities(
            self.G, self.current_path, STRATEGIES[strategy],
            params["p"], params["q"])

        # 3. Actualizar todas las vistas
        self._refresh_graph()
        self._refresh_probs()
        self._update_stats()
        self._log_walk(self.current_path, strategy, params["start"])

    def _step_walk(self):
        """
        Avanza UN solo paso sobre la caminata activa.

        Invariante mantenido:
            current_probs siempre contiene P(salto) DESDE path[-1],
            es decir, desde el nodo en el que el caminante SE ENCUENTRA AHORA.
        """
        params = self._get_params()
        if params is None:
            return
        strategy = self.strategy_var.get()

        # Inicializar si no hay caminata activa
        if not self.current_path:
            self.current_path = [params["start"]]
            self._log_line(
                f"\n── Paso a paso desde {params['start']} [{strategy}] ──\n",
                "header")

        # Nodo previo (t) y nodo actual (v) para calcular probabilidades
        source  = self.current_path[-2] if len(self.current_path) >= 2 else self.current_path[-1]
        current = self.current_path[-1]

        # Calcular probabilidades de salto DESDE el nodo actual
        probs = STRATEGIES[strategy](
            self.G, source, current,
            p=params["p"], q=params["q"])

        if not probs:
            self._log_line(
                f"  → Nodo {current} sin vecinos. Caminata terminada.\n", "end")
            return

        # Elegir el siguiente nodo según las probabilidades
        nxt = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=1)[0]
        self.current_path.append(nxt)

        step = len(self.current_path) - 1
        self._log_line(
            f"  Paso {step:3d}: {current} → {nxt}  (P={probs[nxt]:.4f})\n",
            "step")

        # ─── CLAVE: recalcular current_probs DESDE el nuevo nodo actual (nxt)
        # Así la tab "Probabilidades" siempre muestra "¿a dónde puedo ir desde aquí?"
        self.current_probs = step_probabilities(
            self.G, self.current_path, STRATEGIES[strategy],
            params["p"], params["q"])

        self._refresh_graph()
        self._refresh_probs()
        self._update_stats()

    def _run_compare(self):
        """
        Ejecuta las 4 estrategias con los mismos parámetros y rellena
        la pestaña de comparación.

        Nota sobre las barras de probabilidad:
            Cada estrategia termina en un nodo diferente, por lo que las
            barras muestran P(salto) desde el nodo final de cada caminata.
            Esto permite comparar cómo cada estrategia "ve" su entorno local.
        """
        params = self._get_params()
        if params is None:
            return
        start, stop_prob = params["start"], params["stop_prob"]
        p, q, max_steps  = params["p"], params["q"], params["max_steps"]

        strategies = list(STRATEGIES.keys())
        paths:  dict = {}
        probes: dict = {}   # P(salto) desde el nodo final de cada caminata

        for strat in strategies:
            path = walk(self.G, start, STRATEGIES[strat], stop_prob, p, q, max_steps)
            paths[strat]  = path
            probes[strat] = step_probabilities(self.G, path, STRATEGIES[strat], p, q)

        # ── Figura 2 × 4: fila 0 = redes, fila 1 = barras de probabilidad ──
        self.cmp_fig.clear()
        self.cmp_fig.set_facecolor("#0D1117")

        for col, strat in enumerate(strategies):
            color  = STRATEGY_COLORS[strat]
            path   = paths[strat]
            prob   = probes[strat]
            vc     = Counter(path)
            visited = set(path)
            cov    = len(visited) / self.G.number_of_nodes() * 100

            # ── Red (fila 0) ──
            ax_net = self.cmp_fig.add_subplot(2, 4, col + 1)
            ax_net.set_facecolor("#0D1117")

            nc_map = {n: "#263238" for n in self.G.nodes()}
            ns_map = {n: 160       for n in self.G.nodes()}
            if vc:
                mx  = max(vc.values())
                r_  = int(color[1:3], 16)
                g_  = int(color[3:5], 16)
                b_  = int(color[5:7], 16)
                for node, cnt in vc.items():
                    inten       = 0.3 + 0.7 * (cnt / mx)
                    nc_map[node] = "#{:02x}{:02x}{:02x}".format(
                        int(r_ * inten), int(g_ * inten), int(b_ * inten))
                    ns_map[node] = 160 + 280 * (cnt / mx)

            nc_map[start]    = "#F9A825"
            ns_map[start]    = 480
            nc_map[path[-1]] = "#EF5350"
            ns_map[path[-1]] = 480

            we  = {(min(path[i], path[i+1]), max(path[i], path[i+1]))
                   for i in range(len(path) - 1)}
            ec  = [color if (min(u,v),max(u,v)) in we else "#263238"
                   for u, v in self.G.edges()]
            ew  = [2.0  if (min(u,v),max(u,v)) in we else 0.5
                   for u, v in self.G.edges()]

            nx.draw_networkx_edges(self.G, self.pos, ax=ax_net,
                                   edge_color=ec, width=ew, alpha=0.85)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax_net,
                                   node_color=[nc_map[n] for n in self.G.nodes()],
                                   node_size =[ns_map[n] for n in self.G.nodes()],
                                   linewidths=0.7, edgecolors="#37474F")
            nx.draw_networkx_labels(self.G, self.pos,
                                    {n: str(n) for n in self.G.nodes()},
                                    ax=ax_net, font_color="white",
                                    font_size=5, font_weight="bold")
            ax_net.set_title(
                f"{strat}\n"
                f"len={len(path)}  únicos={len(visited)}  cob={cov:.0f}%",
                color=color, fontsize=7, fontweight="bold", pad=4)
            ax_net.axis("off")

            # ── Barras de P(salto) desde el nodo final (fila 1) ──
            ax_bar = self.cmp_fig.add_subplot(2, 4, col + 5)
            ax_bar.set_facecolor("#0D1117")

            if prob:
                nbs  = [str(n) for n in prob]
                pvs  = list(prob.values())
                bars = ax_bar.bar(nbs, pvs, color=color,
                                  edgecolor="#0D1117", linewidth=0.4)
                max_pv = max(pvs)
                for bar, pv in zip(bars, pvs):
                    if pv >= max_pv * 0.1:
                        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                    pv + max_pv * 0.02,
                                    f"{pv:.2f}", ha="center", va="bottom",
                                    color="white", fontsize=6)
                ax_bar.set_ylim(0, max_pv * 1.3)
            else:
                ax_bar.text(0.5, 0.5, "sin vecinos",
                            ha="center", va="center",
                            color="#546E7A", fontsize=8,
                            transform=ax_bar.transAxes)

            # ── Título: node2vec es 2.º orden → mostrar también el nodo fuente (t)
            if strat == "Node2Vec RW":
                src_node = path[-2] if len(path) >= 2 else path[-1]
                bar_title = f"P(→) t={src_node}→{path[-1]}\n(p={p:.2f}, q={q:.2f})"
            else:
                bar_title = f"P(→) desde nodo {path[-1]}"

            ax_bar.set_title(bar_title, color=color, fontsize=7, pad=3)
            ax_bar.set_xlabel("Vecino", color="#546E7A", fontsize=6)
            ax_bar.set_ylabel("Prob.",  color="#546E7A", fontsize=6)
            ax_bar.tick_params(axis="both", colors="#546E7A", labelsize=6)
            ax_bar.tick_params(axis="x", rotation=45)
            ax_bar.spines[:].set_color("#263238")

        self.cmp_fig.suptitle(
            f"Comparación  ·  inicio: {start}  ·  "
            f"P(parada)={stop_prob:.2f}  ·  p={p:.2f}  q={q:.2f}",
            color="#80CBC4", fontsize=9, fontweight="bold")
        self.cmp_fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.cmp_canvas.draw()

        # ── Tabla de métricas ──
        hdr  = (f"{'Estrategia':<22} {'Longitud':>9} "
                f"{'Únicos':>7} {'Cobertura':>10} {'Nodo final':>11}\n")
        sep  = "─" * 63 + "\n"
        rows = []
        for strat in strategies:
            path = paths[strat]
            vis  = set(path)
            cov  = len(vis) / self.G.number_of_nodes() * 100
            rows.append(
                f"{strat:<22} {len(path):>9} {len(vis):>7} "
                f"{cov:>9.1f}% {path[-1]:>11}")

        self.cmp_metrics.config(state="normal")
        self.cmp_metrics.delete("1.0", "end")
        self.cmp_metrics.insert("end", hdr + sep + "\n".join(rows) + "\n")
        self.cmp_metrics.config(state="disabled")

        self.notebook.select(2)   # saltar a la pestaña de comparación

        # Log
        self._log_line(
            f"\n══ COMPARACIÓN  [{self.graph_var.get()}]  "
            f"inicio={start}  stop={stop_prob:.2f}  "
            f"p={p:.2f}  q={q:.2f}  max={max_steps} ══\n",
            "compare")
        for strat in strategies:
            path = paths[strat]
            vis  = set(path)
            cov  = len(vis) / self.G.number_of_nodes() * 100
            self._log_line(
                f"  {strat:<22}  len={len(path):3d}  "
                f"únicos={len(vis):3d}  cob={cov:.1f}%\n", "normal")

    def _clear(self):
        self._clear_state()
        self._refresh_graph()
        self._draw_probs_empty("Caminata limpiada")
        self._update_stats()
        self._log_line("── Caminata limpiada ──\n", "header")

    # ═══════════════════════════════════════════════════════════════════════
    #  ESTADO DE p/q (solo activos en Node2Vec)
    # ═══════════════════════════════════════════════════════════════════════

    def _update_pq_state(self):
        is_n2v = self.strategy_var.get() == "Node2Vec RW"
        for entry in (self.entry_p, self.entry_q):
            if is_n2v:
                entry.config(state="normal",
                             bg=ENTRY_KW["bg"], fg=ENTRY_KW["fg"],
                             highlightbackground=ENTRY_KW["highlightbackground"])
            else:
                entry.config(state="disabled",
                             disabledbackground=DIS_BG,
                             disabledforeground=DIS_FG)

    def _on_strategy_change(self):
        self._update_pq_state()
        self._update_formula()
        self._refresh_graph()

    # ═══════════════════════════════════════════════════════════════════════
    #  RENDERS / ACTUALIZACIONES DE VISTAS
    # ═══════════════════════════════════════════════════════════════════════

    def _refresh_graph(self):
        """Redibuja la tab Red con la caminata activa."""
        strategy   = self.strategy_var.get()
        start_node = None
        try:
            start_node = int(self.start_var.get())
        except ValueError:
            pass

        draw_network(
            self.ax, self.G, self.pos,
            path          = self.current_path,
            start_node    = start_node,
            strategy      = strategy,
            show_probs    = self.show_probs_var.get(),
            current_probs = self.current_probs,   # siempre desde path[-1]
        )

        patches = [
            mpatches.Patch(color="#F9A825", label="Nodo inicio"),
            mpatches.Patch(color="#EF5350", label="Nodo actual"),
            mpatches.Patch(color=STRATEGY_COLORS.get(strategy, "#4FC3F7"),
                           label=f"Visitados ({strategy})"),
        ]
        self.ax.legend(handles=patches, loc="upper left",
                       facecolor="#1E2A38", edgecolor="#37474F",
                       labelcolor="white", fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_probs_empty(self, msg: str = "Sin caminata activa"):
        """Muestra un mensaje placeholder en la tab de probabilidades."""
        self.prob_ax.clear()
        self.prob_ax.set_facecolor("#0D1117")
        self.prob_fig.set_facecolor("#0D1117")
        self.prob_ax.text(0.5, 0.5, msg, ha="center", va="center",
                          color="#546E7A", fontsize=12,
                          transform=self.prob_ax.transAxes)
        self.prob_canvas.draw()

    def _refresh_probs(self):
        """
        Actualiza la tab Probabilidades.
        Siempre muestra P(salto) DESDE el nodo actual (path[-1]).
        """
        if not self.current_probs or not self.current_path:
            self._draw_probs_empty()
            return

        self.prob_ax.clear()
        self.prob_ax.set_facecolor("#0D1117")
        self.prob_fig.set_facecolor("#0D1117")

        cur_node  = self.current_path[-1]
        strategy  = self.strategy_var.get()
        color     = STRATEGY_COLORS.get(strategy, "#4FC3F7")
        neighbors = [str(n) for n in self.current_probs]
        probs     = list(self.current_probs.values())

        bars = self.prob_ax.bar(neighbors, probs, color=color,
                                edgecolor="#0D1117", linewidth=0.5)
        for bar, pv in zip(bars, probs):
            self.prob_ax.text(
                bar.get_x() + bar.get_width() / 2,
                pv + 0.005,
                f"{pv:.3f}", ha="center", va="bottom",
                color="white", fontsize=8)

        self.prob_ax.set_title(
            f"P(salto) desde nodo {cur_node}  [{strategy}]",
            color="white", fontsize=10)
        self.prob_ax.set_xlabel("Vecino destino", color="#90A4AE")
        self.prob_ax.set_ylabel("Probabilidad",   color="#90A4AE")
        self.prob_ax.tick_params(colors="#90A4AE")
        self.prob_ax.set_ylim(0, max(probs) * 1.25)
        self.prob_ax.spines[:].set_color("#263238")
        self.prob_fig.tight_layout()
        self.prob_canvas.draw()

    def _update_stats(self):
        """Actualiza el panel de estadísticas."""
        G    = self.G
        path = self.current_path
        vis  = set(path)

        lines = [
            f"Nodos:          {G.number_of_nodes()}",
            f"Aristas:        {G.number_of_edges()}",
            f"Densidad:       {nx.density(G):.4f}",
            f"Grado medio:    {np.mean([d for _, d in G.degree()]):.2f}",
            "──────────────────────",
            f"Long. caminata: {len(path)}",
            f"Nodos únicos:   {len(vis)}",
            f"Cobertura:      {len(vis)/G.number_of_nodes()*100:.1f}%",
        ]
        if path:
            lines.append(f"Nodo actual:    {path[-1]}")

        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("end", "\n".join(lines))
        self.stats_text.config(state="disabled")

    def _update_formula(self):
        """Muestra la fórmula de la estrategia activa en el panel lateral."""
        formulas = {
            "Traditional RW":
                "P(v→u) = 1 / deg(v)\n\n"
                "Uniforme sobre todos\nlos vecinos de v.",
            "Degree-Biased RW":
                "P(v→u) = deg(u) /\n"
                "  Σ_{w∈N(v)} deg(w)\n\n"
                "Favorece vecinos con\nmayor grado (hubs).",
            "Inverse-Degree RW":
                "P(v→u) = (1/deg(u)) /\n"
                "  Σ_{w∈N(v)} 1/deg(w)\n\n"
                "Favorece vecinos con\nmenor grado (periferia).",
            "Node2Vec RW":
                "α(t,x) = 1/p  si x == t\n"
                "       = 1    si x ∈ N(t)\n"
                "       = 1/q  si x ∉ N(t)\n\n"
                "P(v→x|t) = α(t,x) / Z\n"
                "Markov de 2.º orden.",
        }
        txt = formulas.get(self.strategy_var.get(), "")
        self.formula_text.config(state="normal")
        self.formula_text.delete("1.0", "end")
        self.formula_text.insert("end", txt)
        self.formula_text.config(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════
    #  LOG
    # ═══════════════════════════════════════════════════════════════════════

    def _log_walk(self, path: list, strategy: str, start: int):
        self._log_line(f"\n══ Caminata [{strategy}] ══\n", "header")
        self._log_line(f"  Inicio:       ", "normal")
        self._log_line(f"{start}\n", "start")
        self._log_line(f"  Longitud:     {len(path)}\n", "normal")
        self._log_line(f"  Nodos únicos: {len(set(path))}\n", "normal")
        self._log_line(f"  Secuencia:\n  ", "normal")

        chunk = []
        for i, n in enumerate(path):
            chunk.append(str(n))
            if (i + 1) % 12 == 0:
                self._log_line(" → ".join(chunk) + "\n  ", "normal")
                chunk = []
        if chunk:
            self._log_line(" → ".join(chunk) + "\n", "normal")

        self._log_line("  Fin:          ", "normal")
        self._log_line(f"{path[-1]}\n", "end")

    def _log_line(self, text: str, tag: str = "normal"):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text, tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()