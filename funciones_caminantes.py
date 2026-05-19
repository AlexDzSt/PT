import random
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
#  Cada función recibe el nodo actual y la red, y devuelve el nodo siguiente.
#  Devuelven None si el nodo no tiene vecinos.
# ─────────────────────────────────────────────────────────────────────────────


def step_traditional(current, G: nx.Graph) -> int | None:
    """
    Caminata tradicional.
    Cada vecino tiene la misma probabilidad de ser visitado.

    Parámetros
    ----------
    current : nodo actual en la red G
    G       : grafo NetworkX

    Devuelve
    -------
    Nodo siguiente, o None si current no tiene vecinos.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    return random.choice(neighbors)


def step_degree_biased(current, G: nx.Graph) -> int | None:
    """
    Caminata sesgada por grado.
    La probabilidad de saltar a un vecino es proporcional a su grado:
        P(u) = deg(u) / sum(deg(v) for v in neighbors)

    Parámetros
    ----------
    current : nodo actual en la red G
    G       : grafo NetworkX

    Devuelve
    -------
    Nodo siguiente, o None si current no tiene vecinos.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    weights = [G.degree(u) for u in neighbors]
    total = sum(weights)
    if total == 0:
        return random.choice(neighbors)
    return random.choices(neighbors, weights=weights, k=1)[0]


def step_inverse_degree(current, G: nx.Graph) -> int | None:
    """
    Caminata sesgada por grado inverso.
    La probabilidad de saltar a un vecino es proporcional a 1/deg(u):
        P(u) = (1/deg(u)) / sum(1/deg(v) for v in neighbors)

    Parámetros
    ----------
    current : nodo actual en la red G
    G       : grafo NetworkX

    Devuelve
    -------
    Nodo siguiente, o None si current no tiene vecinos.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None
    weights = [1.0 / G.degree(u) for u in neighbors]
    total = sum(weights)
    if total == 0:
        return random.choice(neighbors)
    return random.choices(neighbors, weights=weights, k=1)[0]


def step_node2vec(current, G: nx.Graph, prev=None, p: float = 1.0, q: float = 1.0) -> int | None:
    """
    Caminata node2vec.
    Los parámetros p y q controlan el sesgo de retorno y de exploración:

        alpha(x) = 1/p  si x == prev   (retorno al nodo anterior)
                 = 1    si x es vecino de prev también  (triangulo local)
                 = 1/q  en otro caso   (exploración distante)

    Parámetros
    ----------
    current : nodo actual en la red G
    G       : grafo NetworkX
    prev    : nodo previo; si es None se trata como primer paso
    p       : parámetro de retorno   (p > 1 desincentiva volver atrás)
    q       : parámetro de exploración (q > 1 favorece vecindad local, q < 1 favorece exploración DFS)

    Devuelve
    -------
    Nodo siguiente, o None si current no tiene vecinos.
    """
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None

    # Sin nodo previo: primer paso, distribución uniforme
    if prev is None:
        return random.choice(neighbors)

    prev_neighbors = set(G.neighbors(prev))

    weights = []
    for x in neighbors:
        if x == prev:
            alpha = 1.0 / p
        elif x in prev_neighbors:
            alpha = 1.0
        else:
            alpha = 1.0 / q
        weights.append(alpha)

    return random.choices(neighbors, weights=weights, k=1)[0]