import random
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
#  PROBABILIDADES DE SALTO
# ─────────────────────────────────────────────────────────────────────────────

def jump_prob_traditional(G: nx.Graph, source, current, p: float = 1.0, q: float = 1.0) -> dict:
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return {}
    prob = 1.0 / len(neighbors)
    return {u: prob for u in neighbors}


def jump_prob_degree_biased(G: nx.Graph, source, current, p: float = 1.0, q: float = 1.0) -> dict:
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return {}
    degrees = {u: G.degree(u) for u in neighbors}
    total   = sum(degrees.values())
    if total == 0:
        return jump_prob_traditional(G, source, current)
    return {u: degrees[u] / total for u in neighbors}


def jump_prob_inverse_degree(G: nx.Graph, source, current, p: float = 1.0, q: float = 1.0) -> dict:
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return {}
    inv = {u: 1.0 / G.degree(u) for u in neighbors}
    total = sum(inv.values())
    if total == 0:
        return jump_prob_traditional(G, source, current)
    return {u: inv[u] / total for u in neighbors}


def jump_prob_node2vec(G: nx.Graph, source, current, p: float = 1.0, q: float = 1.0) -> dict:
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return {}

    t      = source
    t_nbrs = set(G.neighbors(t))

    unnorm = {}
    for x in neighbors:
        if x == t:
            alpha = 1.0 / p
        elif x in t_nbrs:
            alpha = 1.0
        else:
            alpha = 1.0 / q
        unnorm[x] = alpha

    total = sum(unnorm.values())
    return {x: unnorm[x] / total for x in unnorm}


# ─────────────────────────────────────────────────────────────────────────────
#  CAMINANTE
# ─────────────────────────────────────────────────────────────────────────────

def walk(
    G: nx.Graph,
    start: int,
    jump_fn,
    stop_prob: float = 0.15,
    p: float = 1.0,
    q: float = 1.0,
    max_steps: int = 200,
) -> list:
    """
    Ejecuta una caminata aleatoria sobre G.

    Parámetros
    ----------
    jump_fn   : callable  funcion de probabilidad de salto, e.g. jump_prob_traditional
    stop_prob : float     probabilidad de detenerse en cada paso
    p, q      : float     parametros de node2vec (ignorados por otras estrategias)
    max_steps : int       numero maximo de pasos
    """
    path    = [start]
    source  = start
    current = start

    for _ in range(max_steps):
        if random.random() < stop_prob:
            break

        probs = jump_fn(G, source, current, p=p, q=q)
        if not probs:
            break

        nodes   = list(probs.keys())
        weights = list(probs.values())
        nxt     = random.choices(nodes, weights=weights, k=1)[0]

        path.append(nxt)
        source  = current
        current = nxt

    return path


def step_probabilities(
    G: nx.Graph,
    path: list,
    jump_fn,
    p: float = 1.0,
    q: float = 1.0,
) -> dict:
    """
    Retorna P(salto) desde el nodo actual (path[-1]).
    Para estrategias de 2o orden (node2vec) usa path[-2] como fuente.
    """
    if not path:
        return {}
    current = path[-1]
    source  = path[-2] if len(path) >= 2 else path[-1]
    return jump_fn(G, source, current, p=p, q=q)