"""
datos_caminata.py
─────────────────
Funciones puras que calculan métricas sobre una caminata aleatoria.
No dependen de ninguna librería de visualización.

Parámetros comunes
──────────────────
path         : list  – secuencia de nodos visitados (incluye el nodo inicial)
total_nodos  : int   – número total de nodos del grafo
"""


def longitud(path: list) -> int:
    """Número de pasos de la caminata (= len(path))."""
    return len(path)


def nodos_unicos(path: list) -> int:
    """Cantidad de nodos distintos visitados."""
    return len(set(path))


def cobertura(path: list, total_nodos: int) -> float:
    """
    Fracción de nodos del grafo visitados al menos una vez.
    Retorna un valor en [0, 1].
    """
    return len(set(path)) / total_nodos


def nodo_actual(path: list):
    """
    Último nodo de la caminata (posición actual del caminante).
    Retorna None si la caminata está vacía.
    """
    return path[-1] if path else None


def curva_cobertura(path: list, total_nodos: int) -> list[tuple[float, float]]:
    """
    Genera la curva de cobertura paso a paso.

    Retorna una lista de tuplas (longitud_relativa, cobertura) donde:
      · longitud_relativa = (índice de paso + 1) / total_nodos
      · cobertura         = nodos únicos vistos hasta ese paso / total_nodos

    Ejemplo:
        path = [0, 1, 0, 2], total_nodos = 4
        → [(0.25, 0.25), (0.5, 0.5), (0.75, 0.5), (1.0, 0.75)]
    """
    visto  = set()
    puntos = []
    for i, nodo in enumerate(path):
        visto.add(nodo)
        puntos.append((
            (i + 1) / total_nodos,   # longitud relativa
            len(visto) / total_nodos  # cobertura
        ))
    return puntos
